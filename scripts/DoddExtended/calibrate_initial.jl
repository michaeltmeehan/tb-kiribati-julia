
using TBKiribatiJulia
using Random, CSV, DataFrames
using Distributions: Normal, truncated, median
using Optim

include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))

# --------------------------------------------------
# Configuration
# --------------------------------------------------

start_year = 1970
end_year = 2026

setup = setup_dodd_analysis(
    start_year=start_year,
    end_year=end_year,
    output_years=2013:2025,
)

presentation = load_kiribati_presentation(
    setup.surveillance_dir;
    years=2007:2012,
)

model = DoddExtendedTBModel(
    setup.model.contact,
    presentation.presentation;
    reinfection=true,
)

# --------------------------------------------------
# Observations
# --------------------------------------------------

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

recurrence = RecurrentTBObservation(
    filter(
        r -> 1996 <= r.year <= 2024,
        setup.surveillance.recurrent_notifications,
    ),
    national,
)

site = SiteTBObservation(filter(
    r -> start_year <= r.year < end_year,
    setup.surveillance.site_notifications,
))

deaths = TreatmentDeathObservation(
    filter(
        r -> 2012 <= r.year <= 2023,
        setup.surveillance.treatment_outcomes,
    ),
)

observation_model = AnnualRandomEffectNB2Observation(
    truncated(Normal(0.0, 0.3), 0.0, Inf),
    truncated(Normal(0.0, 0.3), 0.0, Inf);
    quadrature_order=100,
)

problem = CalibrationProblem(
    model,
    setup.demography,
    notifications;
    start_year=start_year,
    end_year=end_year,
    age_breaks=setup.age_breaks,
    observation_model=observation_model,
    treatment_deaths=deaths,
    recurrence=recurrence,
    site=site,
)

# --------------------------------------------------
# Free and fixed parameters
# --------------------------------------------------

free_names = (
    :beta,
    :psi,
    :Lambda0,
    :epsilon,
    :nu,
    :omega,
    :detection_logitK,
    :extrapulmonary_detection,
    :sigma_obs,
    :tau_year,
)

# Build a complete parameter vector from the new priors.
initial = map(median, problem.prior)

# Explicit starting values informed by the previous
# F_nu_only calibration.
#
# These are starting points, not fixed estimates.
initial = merge(
    initial,
    (
        beta=10.83,
        psi=0.1442,
        epsilon=0.04675,
        nu=0.01454,
        omega=0.004337,
        detection_logitK=log(0.409575 / (1 - 0.409575)),
        extrapulmonary_detection=0.72805,
        sigma_obs=0.31056,
        tau_year=0.18286,
    ),
)

# Fix parameters not selected for estimation.
fixed_names = Tuple(
    k for k in keys(problem.prior)
          if !(k in free_names)
)

fixed = NamedTuple{fixed_names}(
    Tuple(getproperty(initial, k) for k in fixed_names)
)

# Apply explicit fixed assumptions.
fixed = merge(
    fixed,
    (
        kappa=1.85893,
        duration_sp=1.6,
        duration_sn=5.35,
        duration_ep=3.0,
        treatment_cfr=304 / 4928,
        detection_trend=0.0,
        untreated_cfr_sp=0.70,
        untreated_cfr_sn=0.20,
        untreated_cfr_ep=0.20,
        smear_negative_infectiousness=0.35,
        smear_negative_detection=1.0,
    ),
)

@assert all(k -> !haskey(fixed, k), free_names)
@assert all(k -> haskey(problem.prior, k), keys(fixed))

initial = merge(initial, fixed)

println("\nInitial CFR parameters:")
println((
    sp=initial.untreated_cfr_sp,
    sn=initial.untreated_cfr_sn,
    ep=initial.untreated_cfr_ep,
))

# --------------------------------------------------
# Likelihood decomposition
# --------------------------------------------------

function likelihood_components(problem, theta)
    out = predict(problem, theta)

    lnotif = notification_loglikelihood(
        problem.observation_model,
        problem.observations,
        out,
        problem.age_breaks,
        theta,
    )

    lrec = recurrence_loglikelihood(
        problem.recurrence, out
    )

    lsite = site_loglikelihood(
        problem.site, out
    )

    ldeath = treatment_death_loglikelihood(
        problem.treatment_deaths, theta
    )

    return (
        notifications=lnotif,
        recurrence=lrec,
        site=lsite,
        treatment_deaths=ldeath,
        total=lnotif + lrec + lsite + ldeath,
    )
end

println("\nInitial likelihood:")
display(likelihood_components(problem, initial))

# --------------------------------------------------
# Initial MAP fit
# --------------------------------------------------

fit = fit_map(
    problem;
    fixed=fixed,
    initial=initial,
    optimizer=:nelder_mead,
    nstarts=1,
    iterations=3000,
    rng=MersenneTwister(42),
)

println("\nConverged: ", fit.converged)
println("Log posterior: ", fit.log_posterior)

println("\nFitted likelihood:")
display(likelihood_components(problem, fit.parameters))

# --------------------------------------------------
# Parameter comparison
# --------------------------------------------------

parameters = DataFrame(
    parameter=String.(collect(keys(problem.prior))),
    initial=Float64.(collect(values(initial))),
    fitted=Float64.(collect(values(fit.parameters))),
    free=[k in free_names for k in keys(problem.prior)],
)

parameters.relative_change = [
    iszero(r.initial) ? missing :
    r.fitted / r.initial - 1
    for r in eachrow(parameters)
]

display(parameters)

# --------------------------------------------------
# Diagnostic predictions
# --------------------------------------------------

output = predict(
    problem,
    fit.parameters;
    variables=nothing,
)

diagnostics = DataFrame(
    year=output.years,
    notifications=vec(sum(output.notifications; dims=1)),
    recurrent=vec(sum(output.recurrent_notifications; dims=1)),
    pulmonary=vec(sum(output.pulmonary_notifications; dims=1)),
    extrapulmonary=vec(sum(output.extrapulmonary_notifications; dims=1)),
    incidence=vec(sum(output.incidence; dims=1)),
    mortality=vec(sum(output.mortality; dims=1)),
)

diagnostics.recurrent_fraction =
    diagnostics.recurrent ./ diagnostics.notifications

diagnostics.pulmonary_fraction =
    diagnostics.pulmonary ./ diagnostics.notifications

println("Log likelihood: ", fit.log_likelihood)
println("Log prior: ", fit.log_prior)
println("Log posterior: ", fit.log_posterior)

println("Optimiser iterations: ",
    Optim.iterations(fit.optim_result))

# --------------------------------------------------
# Save results
# --------------------------------------------------

results_dir = normpath(joinpath(
    @__DIR__, "..", "..",
    "results", "DoddExtended", "current_calibration",
))

mkpath(results_dir)

CSV.write(
    joinpath(results_dir, "parameters.csv"),
    parameters,
)

CSV.write(
    joinpath(results_dir, "predictions.csv"),
    diagnostics,
)

components = likelihood_components(problem, fit.parameters)

CSV.write(
    joinpath(results_dir, "likelihood.csv"),
    DataFrame(
        component=String.(collect(keys(components))),
        loglikelihood=Float64.(collect(values(components))),
    ),
)

println("\nResults saved to: ", results_dir)
