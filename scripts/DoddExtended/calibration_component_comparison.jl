
using TBKiribatiJulia
using Random, CSV, DataFrames
using Distributions: Normal, truncated, median

const TB = TBKiribatiJulia

include(joinpath(
    @__DIR__, "..", "common", "kiribati_setup.jl"
))

# --------------------------------------------------
# Configuration
# --------------------------------------------------

start_year = 1970
end_year = 2026

result_dir = joinpath(
    "results", "DoddExtended",
    "calibration_component_comparison"
)
mkpath(result_dir)

setup = setup_dodd_analysis(
    start_year=start_year,
    end_year=end_year,
    output_years=2013:2024,
)

presentation = TB.load_kiribati_presentation(
    setup.surveillance_dir;
    years=2007:2012,
)

model = TB.DoddExtendedTBModel(
    setup.contact,
    presentation.presentation;
    reinfection=true,
)

# --------------------------------------------------
# Observational datasets
# --------------------------------------------------

national = filter(
    r -> 1980 <= r.year <= 2025,
    setup.surveillance.national_notifications,
)

notifications = TB.combine_notification_observations(
    setup.age_observations,
    national;
    n_age_groups=length(setup.age_labels),
    start_year=start_year,
    end_year=end_year,
)

recurrence = TB.RecurrentTBObservation(
    filter(
        r -> 1996 <= r.year <= 2024,
        setup.surveillance.recurrent_notifications,
    ),
    national,
)

site = TB.SiteTBObservation(filter(
    r -> 2013 <= r.year <= 2025,
    setup.surveillance.site_notifications,
))

observation_model = TB.AnnualRandomEffectNB2Observation(
    truncated(Normal(0.0, 0.3), 0.0, Inf),
    truncated(Normal(0.0, 0.3), 0.0, Inf);
    quadrature_order=40,
)

# --------------------------------------------------
# Construct calibration problems
# --------------------------------------------------

function make_problem(; use_recurrence, use_site)
    return TB.CalibrationProblem(
        model,
        setup.demography,
        notifications;
        start_year=start_year,
        end_year=end_year,
        age_breaks=setup.age_breaks,
        observation_model=observation_model,
        recurrence=use_recurrence ? recurrence : nothing,
        site=use_site ? site : nothing,
    )
end

problems = [
    make_problem(
        use_recurrence=false,
        use_site=false,
    ),
    make_problem(
        use_recurrence=true,
        use_site=false,
    ),
    make_problem(
        use_recurrence=true,
        use_site=true,
    ),
]

names = ("A", "B", "C")

free_names = (
    :beta,
    :Lambda0,
    :detection_logitK,
    :detection_trend,
    :omega,
)

initial = map(median, problems[1].prior)

# Fixed observation dispersion.
initial = merge(
    initial,
    (
        omega=0.001,
        sigma_obs=0.30,
        tau_year=0.18,
    ),
)

fixed_names = Tuple(
    k for k in keys(problems[1].prior)
          if !(k in free_names)
)

fixed = NamedTuple{fixed_names}(
    Tuple(initial[k] for k in fixed_names)
)

@assert all(k -> haskey(problems[1].prior, k),
    free_names)

# --------------------------------------------------
# Fit all three experiments
# --------------------------------------------------

fits = []

for i in eachindex(problems)
    problem = problems[i]
    name = names[i]

    # Use previous fit as a warm start.
    # Other starts are independently sampled.
    start = i == 1 ? initial : fits[end].parameters

    println("\n", "="^50)
    println("EXPERIMENT $name")
    println("="^50)

    fit = TB.fit_mle(
        problem;
        fixed=fixed,
        initial=start,
        optimizer=:nelder_mead,
        iterations=500,
        nstarts=1,
        rng=MersenneTwister(42 + i),
    )

    push!(fits, fit)

    println("Converged: ", fit.converged)
    println("Log likelihood: ", fit.log_likelihood)
    println("Log prior: ", fit.log_prior)

    for k in free_names
        println(k, " = ", fit.parameters[k])
    end

    CSV.write(
        joinpath(result_dir, "parameters_$name.csv"),
        DataFrame(
            parameter=string.(collect(keys(fit.parameters))),
            value=Float64.(collect(values(fit.parameters))),
        ),
    )
end

# --------------------------------------------------
# Evaluate all fitted models against ALL datasets
# --------------------------------------------------

# Important:
# Evaluate each fitted parameter vector against the
# SAME full likelihood, even when some components
# were excluded during optimisation.

full_problem = problems[3]

comparison = NamedTuple[]
notification_rows = DataFrame()

for (i, fit) in enumerate(fits)
    name = names[i]
    theta = fit.parameters

    out = TB.predict(
        full_problem,
        theta;
        variables=(
            :population,
            :incidence,
            :notifications,
            :mortality,
            :untreated_active_prevalence,
            :latent_prevalence,
            :primary_infections,
            :reinfections,
            :fast_progression,
            :slow_progression,
            :relapse,
        ),
    )

    ll_notifications = TB.notification_loglikelihood(
        full_problem.observation_model,
        full_problem.observations,
        out,
        full_problem.age_breaks,
        theta,
    )

    ll_recurrence = TB.recurrence_loglikelihood(
        full_problem.recurrence, out
    )

    ll_site = TB.site_loglikelihood(
        full_problem.site, out
    )

    j = findfirst(==(2023), out.years)
    @assert !isnothing(j)

    total(v) = sum(getproperty(out, v)[:, j])

    pop = total(:population)
    inc = total(:incidence)
    notif = total(:notifications)
    reinf = total(:reinfections)

    push!(comparison, (
        experiment=name,
        converged=fit.converged,
        beta=theta.beta,
        Lambda0=theta.Lambda0,
        detection_logitK=theta.detection_logitK,
        detection_trend=theta.detection_trend,
        log_prior=fit.log_prior,
        notification_ll=ll_notifications,
        recurrence_ll=ll_recurrence,
        site_ll=ll_site,
        common_loglikelihood=(
            ll_notifications + ll_recurrence + ll_site
        ),
        incidence_2023=inc,
        notifications_2023=notif,
        mortality_2023=total(:mortality),
        incidence_per_100k=1e5 * inc / pop,
        active_per_100k=1e5 *
            total(:untreated_active_prevalence) / pop,
        latent_fraction=total(:latent_prevalence) / pop,
        reinfections_2023=reinf,
        reinfections_per_capita=reinf / pop,
        fast_fraction=total(:fast_progression) / inc,
    ))

    rows = DataFrame(
        TB.notification_fit_rows(full_problem, out)
    )
    rows.experiment .= name

    append!(
        notification_rows,
        rows;
        cols=:union,
    )
end

results = DataFrame(comparison)

CSV.write(
    joinpath(result_dir, "comparison.csv"),
    results,
)

CSV.write(
    joinpath(result_dir, "notification_fit.csv"),
    notification_rows,
)

println("\n\nPARAMETER COMPARISON")
display(select(
    results,
    :experiment,
    :converged,
    :beta,
    :Lambda0,
    :detection_logitK,
    :detection_trend,
))

println("\nLIKELIHOOD COMPARISON")
display(select(
    results,
    :experiment,
    :notification_ll,
    :recurrence_ll,
    :site_ll,
    :common_loglikelihood,
))

println("\nEPIDEMIOLOGICAL COMPARISON")
display(select(
    results,
    :experiment,
    :incidence_2023,
    :notifications_2023,
    :mortality_2023,
    :active_per_100k,
    :reinfections_per_capita,
    :fast_fraction,
))

println("\n2023 AGE-SPECIFIC NOTIFICATIONS")
display(filter(
    row -> row.year == 2023,
    notification_rows,
))

println("\nResults saved to: ", result_dir)
