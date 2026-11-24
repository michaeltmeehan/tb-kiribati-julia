using TBKiribatiJulia
using Random, CSV, DataFrames
using Distributions: Normal, truncated, median
using Optim

# Standalone notification-only, low-relapse MAP calibration.
# Run from the repository root:
#   julia --project=. scripts/DoddExtended/notification_only_low_relapse_map.jl

TB = TBKiribatiJulia

include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))

start_year = 1970
end_year = 2026
relapse_hazard = 0.001
max_iterations = 250
result_dir = joinpath("results", "DoddExtended", "notification_only_low_relapse_map")
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

observation_model = TB.AnnualRandomEffectNB2Observation(
    truncated(Normal(0.0, 0.3), 0.0, Inf),
    truncated(Normal(0.0, 0.3), 0.0, Inf);
    quadrature_order=40,
)

# Use the established, robust explicit solver for this baseline.
# This avoids the Rosenbrock/ForwardDiff workspace incompatibility.
problem = TB.CalibrationProblem(
    model,
    setup.demography,
    notifications;
    start_year=start_year,
    end_year=end_year,
    age_breaks=setup.age_breaks,
    observation_model=observation_model,
    solver=TB.Tsit5(),
)

free_names = (
    :beta,
    :Lambda0,
    :detection_logitK,
    :detection_trend,
)

initial = merge(
    map(median, problem.prior),
    (
        beta=30.0,
        Lambda0=0.03,
        detection_logitK=0.0,
        detection_trend=0.0,
        omega=relapse_hazard,
        sigma_obs=0.30,
        tau_year=0.18,
    ),
)

fixed_names = Tuple(k for k in keys(problem.prior) if !(k in free_names))
fixed = NamedTuple{fixed_names}(Tuple(initial[k] for k in fixed_names))

println("\nNOTIFICATION-ONLY LOW-RELAPSE MAP")
println("Fixed omega = ", relapse_hazard, " / year")
println("Solver: Tsit5; iterations: ", max_iterations, "; starts: 1")
println("Initial beta = ", initial.beta, ", Lambda0 = ", initial.Lambda0)

println("\nWarm-up likelihood:")
@time println("log likelihood = ", TB.calibration_loglikelihood(problem, initial))

fit = TB.fit_map(
    problem;
    fixed=fixed,
    initial=initial,
    optimizer=:nelder_mead,
    iterations=max_iterations,
    nstarts=1,
    rng=MersenneTwister(42),
)

println("\nFIT")
println("Converged: ", fit.converged)
println("Iterations: ", Optim.iterations(fit.optim_result))
println("Objective evaluations: ", Optim.f_calls(fit.optim_result))
println("Log likelihood: ", fit.log_likelihood)
println("Log posterior: ", fit.log_posterior)
for k in (free_names..., :omega)
    println(k, " = ", getproperty(fit.parameters, k))
end

CSV.write(
    joinpath(result_dir, "parameters.csv"),
    DataFrame(
        parameter=string.(collect(keys(fit.parameters))),
        value=Float64.(collect(values(fit.parameters))),
        fitted=[k in free_names for k in keys(fit.parameters)],
    ),
)

out = TB.predict(
    problem,
    fit.parameters;
    variables=(
        :population, :incidence, :notifications, :mortality,
        :untreated_active_prevalence, :latent_prevalence,
        :reinfections, :fast_progression, :slow_progression,
        :relapse, :recurrent_notifications,
    ),
)

annual = DataFrame()
for year in (1980, 2000, 2013, 2019, 2023, 2024, 2025)
    j = findfirst(==(year), out.years)
    isnothing(j) && continue
    total(k) = sum(getproperty(out, k)[:, j])
    pop = total(:population)
    inc = total(:incidence)
    n = total(:notifications)
    active = total(:untreated_active_prevalence)
    recurrent = total(:recurrent_notifications)
    push!(annual, (
        year=year,
        population=pop,
        incidence=inc,
        notifications=n,
        mortality=total(:mortality),
        incidence_per_100k=1e5 * inc / pop,
        active_prevalence=active,
        active_prevalence_percent=100 * active / pop,
        latent_fraction_percent=100 * total(:latent_prevalence) / pop,
        reinfections=total(:reinfections),
        fast_fraction=inc > 0 ? total(:fast_progression) / inc : NaN,
        recurrent_fraction=n > 0 ? recurrent / n : NaN,
    ))
end
CSV.write(joinpath(result_dir, "epidemiological_diagnostics.csv"), annual)

p = TB.make_parameters(model, fit.parameters)
detection = DataFrame(
    year=[1980, 2000, 2013, 2019, 2023, 2024],
    detection_probability=[
        TB.detection_probability(p.detection, y)
        for y in (1980, 2000, 2013, 2019, 2023, 2024)
    ],
)
CSV.write(joinpath(result_dir, "detection_diagnostics.csv"), detection)

notification_rows = DataFrame(TB.notification_fit_rows(problem, out))
CSV.write(joinpath(result_dir, "notification_fit.csv"), notification_rows)

println("\nEPIDEMIOLOGICAL DIAGNOSTICS")
display(annual)
println("\nDETECTION TRAJECTORY (model probability, not case ascertainment)")
display(detection)
println("\n2023 AGE-SPECIFIC NOTIFICATION FIT")
display(filter(r -> r.year == 2023, notification_rows))
println("\nResults saved in: ", result_dir)
