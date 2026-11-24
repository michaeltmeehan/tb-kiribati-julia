
using TBKiribatiJulia
using Random
using CSV, DataFrames
using Distributions: Normal, truncated, median

include(joinpath(
    @__DIR__, "..", "common", "kiribati_setup.jl"
))

# --------------------------------------------------
# Setup and observations
# --------------------------------------------------

start_year = 1970
end_year = 2026

setup = setup_dodd_analysis(
    start_year=start_year,
    end_year=end_year,
    output_years=2013:2024,
)

presentation = load_kiribati_presentation(
    setup.surveillance_dir;
    years=2007:2012,
)

model = DoddExtendedTBModel(
    setup.contact,
    presentation.presentation;
    reinfection=false,
)

# Use age-specific counts when available and
# national totals for the remaining years.
national = filter(
    r -> 1980 <= r.year <= 2025,
    setup.surveillance.national_notifications,
)

notifications = combine_notification_observations(
    setup.age_observations,
    national;
    n_age_groups=length(setup.age_labels),
    start_year=start_year,
    end_year=end_year,
)

# Observed recurrent notifications.
recurrence = RecurrentTBObservation(
    filter(
        r -> 1996 <= r.year <= 2024,
        setup.surveillance.recurrent_notifications,
    ),
    national,
)

# Use site classification data after the historical
# 2007–2012 period used to construct presentation.
site = SiteTBObservation(filter(
    r -> 2013 <= r.year <= 2025,
    setup.surveillance.site_notifications,
))

# Observed deaths among registered treatment cohorts.
deaths = TreatmentDeathObservation(
    filter(
        r -> 2012 <= r.year <= 2023,
        setup.surveillance.treatment_outcomes,
    ),
)

observation_model = AnnualRandomEffectNB2Observation(
    truncated(Normal(0.0, 0.3), 0.0, Inf),
    truncated(Normal(0.0, 0.3), 0.0, Inf);
    quadrature_order=40,
)

problem = CalibrationProblem(
    model,
    setup.demography,
    notifications;
    start_year=start_year,
    end_year=end_year,
    age_breaks=setup.age_breaks,
    observation_model=observation_model,
    recurrence=recurrence,
    site=site,
    treatment_deaths=deaths,
)

# --------------------------------------------------
# Four free epidemiological parameters
# --------------------------------------------------

free_names = (
    :beta,
    :Lambda0,
    :detection_logitK,
    :detection_trend,
)

initial = map(median, problem.prior)

# Keep observation dispersion fixed initially.
initial = merge(
    initial,
    (sigma_obs=0.30, tau_year=0.18),
)

fixed_names = Tuple(
    name for name in keys(problem.prior)
             if !(name in free_names)
)

fixed = NamedTuple{fixed_names}(
    Tuple(initial[name] for name in fixed_names)
)

@assert all(name -> haskey(problem.prior, name),
    free_names)
@assert all(name -> !haskey(fixed, name),
    free_names)

# --------------------------------------------------
# Initial likelihood and optimisation
# --------------------------------------------------

println("Initial log likelihood:")
println(calibration_loglikelihood(problem, initial))

fit = TBKiribatiJulia.fit_map(
    problem;
    fixed=fixed,
    initial=initial,
    optimizer=:nelder_mead,
    iterations=1500,
    nstarts=1,
    rng=MersenneTwister(42),
)

println("\nConverged: ", fit.converged)
println("Log likelihood: ", fit.log_likelihood)

println("\nFitted parameters:")
for name in free_names
    println(name, " = ", fit.parameters[name])
end

# --------------------------------------------------
# Predictions and results
# --------------------------------------------------


# Extract annual epidemiological outcomes.
# Include every year, so all observation years are retained.
out = predict(
    problem,
    fit.parameters;
    variables=(
        :population,
        :incidence,
        :notifications,
        :mortality,
        :untreated_active_prevalence,
        :latent_prevalence,
        :treatment_prevalence,
        :fast_progression,
        :slow_progression,
        :relapse,
    ),
    years=collect(start_year:(end_year-1)),
)


result_dir = joinpath(
    "results", "DoddExtended", "four_parameter_mle"
)
mkpath(result_dir)

# --------------------------------------------------
# Epidemiological diagnostics
# --------------------------------------------------

total(variable) = vec(sum(
    getproperty(out, variable);
    dims=1,
))

diagnostics = DataFrame(
    year=out.years,
    population=total(:population),
    incidence=total(:incidence),
    notifications=total(:notifications),
    mortality=total(:mortality),
    untreated_active=total(:untreated_active_prevalence),
    latent=total(:latent_prevalence),
    on_treatment=total(:treatment_prevalence),
    fast_progression=total(:fast_progression),
    slow_progression=total(:slow_progression),
    relapse=total(:relapse),
)

diagnostics.incidence_per_100k =
    1e5 .* diagnostics.incidence ./ diagnostics.population

diagnostics.active_per_100k =
    1e5 .* diagnostics.untreated_active ./ diagnostics.population

diagnostics.mortality_per_100k =
    1e5 .* diagnostics.mortality ./ diagnostics.population

diagnostics.notification_incidence_ratio =
    diagnostics.notifications ./ diagnostics.incidence

diagnostics.latent_fraction =
    diagnostics.latent ./ diagnostics.population

CSV.write(
    joinpath(result_dir, "epidemiological_diagnostics.csv"),
    diagnostics,
)

println("\nEpidemiological diagnostics:")
display(filter(
    row -> row.year in (1970, 2000, 2013, 2019, 2023, 2025),
    diagnostics[:, [
        :year,
        :incidence,
        :notifications,
        :mortality,
        :incidence_per_100k,
        :active_per_100k,
        :notification_incidence_ratio,
        :latent_fraction,
    ]],
))

parameter_table = DataFrame(
    parameter=string.(collect(keys(fit.parameters))),
    value=Float64.(collect(values(fit.parameters))),
    fitted=[
        k in free_names for k in keys(fit.parameters)
    ],
)

CSV.write(
    joinpath(result_dir, "parameters.csv"),
    parameter_table,
)
# --------------------------------------------------
# Age-specific transmission and progression diagnostics
# --------------------------------------------------

function age_progression_diagnostics(problem, theta; year=2023)
    TB = TBKiribatiJulia

    variables = (
        :population,
        :primary_infections,
        :reinfections,
        :early_latent_prevalence,
        :late_latent_prevalence,
        :fast_progression,
        :slow_progression,
        :relapse,
        :incidence,
        :notifications,
        :untreated_active_prevalence,
        :untreated_mortality,
    )

    output = TB.predict(
        problem,
        theta;
        variables=variables,
    )

    j = findfirst(==(year), output.years)
    isnothing(j) && error("Year $year unavailable")

    breaks = problem.age_breaks
    rows = NamedTuple[]

    for g in 1:(length(breaks)-1)
        lo = breaks[g]
        hi = breaks[g+1] - 1
        label = g == length(breaks) - 1 ?
                "$(lo)+" : "$(lo)-$(hi)"

        value(v) = getproperty(output, v)[g, j]

        pop = value(:population)
        primary = value(:primary_infections)
        reinfections = value(:reinfections)
        fast = value(:fast_progression)
        slow = value(:slow_progression)
        relapse = value(:relapse)
        incidence = value(:incidence)

        push!(rows, (
            year=year,
            age_group=label,
            population=pop,
            primary_infections=primary,
            reinfections=reinfections,
            primary_infections_per_100k=1e5 * primary / pop,
            early_latent=value(:early_latent_prevalence),
            late_latent=value(:late_latent_prevalence),
            fast_progression=fast,
            slow_progression=slow,
            relapse=relapse,
            incidence=incidence,
            incidence_per_100k=1e5 * incidence / pop,
            notifications=value(:notifications),
            active_prevalence=value(:untreated_active_prevalence),
            untreated_deaths=value(:untreated_mortality),
            fast_fraction=incidence > 0 ? fast / incidence : NaN,
        ))
    end

    result = DataFrame(rows)

    @assert all(isapprox.(
        result.incidence,
        result.fast_progression .+
            result.slow_progression .+
            result.relapse;
        rtol=1e-8,
    ))

    return result
end

age_diag = age_progression_diagnostics(
    problem, fit.parameters; year=2023
)

CSV.write(
    joinpath(result_dir, "age_progression_2023.csv"),
    age_diag,
)

println("\nAGE-SPECIFIC INFECTION PRESSURE:")
display(select(
    age_diag,
    :age_group,
    :population,
    :primary_infections,
    :reinfections,
    :primary_infections_per_100k,
))

println("\nAGE-SPECIFIC PROGRESSION:")
display(select(
    age_diag,
    :age_group,
    :early_latent,
    :late_latent,
    :fast_progression,
    :slow_progression,
    :relapse,
    :incidence,
    :fast_fraction,
))

println("\nAGE-SPECIFIC BURDEN:")
display(select(
    age_diag,
    :age_group,
    :incidence_per_100k,
    :notifications,
    :active_prevalence,
    :untreated_deaths,
))


fit_rows = notification_fit_rows(problem, out)

CSV.write(
    joinpath(result_dir, "notification_fit.csv"),
    DataFrame(fit_rows),
)

println("\nAge-specific notification fit, 2023:")

display(filter(
    row -> row.year == 2023,
    DataFrame(fit_rows),
))

println("\nLikelihood decomposition:")

ln = notification_loglikelihood(
    problem.observation_model,
    problem.observations,
    out,
    problem.age_breaks,
    fit.parameters,
)

lr = recurrence_loglikelihood(
    problem.recurrence, out
)

ls = site_loglikelihood(
    problem.site, out
)

ld = treatment_death_loglikelihood(
    problem.treatment_deaths, fit.parameters
)

display((
    notifications=ln,
    recurrence=lr,
    site=ls,
    treatment_deaths=ld,
    total=ln + lr + ls + ld,
))
