
using TBKiribatiJulia
using DataFrames, CSV, Random, Optim

# --------------------------------------------------
# Configuration
# --------------------------------------------------

problem = problem100
reference = fitted_models["F_nu_only"]
theta0 = reference.parameters

results_dir = normpath(joinpath(
    @__DIR__, "..", "..", "results",
    "DoddExtended", "mortality_release",
))
mkpath(results_dir)

base_free = (
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

experiments = [
    ("H_cfr", (:untreated_cfr,)),
    ("I_duration", (:duration,)),
    ("J_both", (:untreated_cfr, :duration)),
]

function mortality_summary(problem, theta; year=2024)
    # No years argument: predict() retains all
    # required calibration observation years.
    out = predict(
        problem, theta;
        variables=(
            :incidence,
            :notifications,
            :mortality,
            :untreated_mortality,
            :treatment_mortality,
            :self_cure,
        ),
    )

    j = findfirst(==(year), out.years)
    isnothing(j) && error("Missing prediction year $year")

    total(name) = sum(getproperty(out, name)[:, j])

    untreated = total(:untreated_mortality)
    treatment = total(:treatment_mortality)
    deaths = total(:mortality)
    notifications = total(:notifications)
    incidence = total(:incidence)

    @assert isapprox(
        deaths, untreated + treatment;
        rtol=1e-6, atol=1e-6,
    )

    return (
        notifications=notifications,
        incidence=incidence,
        mortality=deaths,
        untreated_deaths=untreated,
        treatment_deaths=treatment,
        self_cure=total(:self_cure),
        mortality_incidence_ratio=deaths / incidence,
    )
end

function fixed_for(problem, theta, free_names)
    names = Tuple(
        k for k in keys(problem.prior)
              if !(k in free_names)
    )

    NamedTuple{names}(
        Tuple(getproperty(theta, k) for k in names)
    )
end

# --------------------------------------------------
# Run experiments
# --------------------------------------------------

summary_rows = NamedTuple[]
parameter_rows = NamedTuple[]

for (i, (name, released)) in enumerate(experiments)

    println("\n", "="^60)
    println("Experiment: ", name)
    println("Released: ", released)
    println("="^60)

    free_names = (base_free..., released...)

    fixed = fixed_for(problem, theta0, free_names)

    @assert all(k -> !haskey(fixed, k), free_names)

    trial = fit_map(
        problem;
        fixed=fixed,
        initial=theta0,
        optimizer=:nelder_mead,
        nstarts=1,
        iterations=3000,
        rng=MersenneTwister(200 + i),
    )

    fitted_models[name] = trial

    pred = mortality_summary(problem, trial.parameters)

    push!(summary_rows, (
        experiment=name,
        converged=trial.converged,
        iterations=Optim.iterations(trial.optim_result),
        loglikelihood=trial.log_likelihood,
        logprior=trial.log_prior,
        logposterior=trial.log_posterior,
        nu=trial.parameters.nu,
        untreated_cfr=trial.parameters.untreated_cfr,
        duration=trial.parameters.duration,
        tau_year=trial.parameters.tau_year,
        pred...,
    ))

    for k in keys(problem.prior)
        push!(parameter_rows, (
            experiment=name,
            parameter=String(k),
            reference=Float64(getproperty(theta0, k)),
            fitted=Float64(getproperty(trial.parameters, k)),
            free=(k in free_names),
        ))
    end

    # Save after every fit.
    CSV.write(
        joinpath(results_dir, "summary.csv"),
        DataFrame(summary_rows),
    )

    CSV.write(
        joinpath(results_dir, "parameters.csv"),
        DataFrame(parameter_rows),
    )

    println("Converged: ", trial.converged)
    println("Log likelihood: ", trial.log_likelihood)
    println("Log posterior: ", trial.log_posterior)
    println("2024: ", pred)
end

# --------------------------------------------------
# Compare with F
# --------------------------------------------------

reference_summary = mortality_summary(
    problem, theta0
)

reference_row = (
    experiment="F_nu_only",
    converged=reference.converged,
    iterations=Optim.iterations(reference.optim_result),
    loglikelihood=reference.log_likelihood,
    logprior=reference.log_prior,
    logposterior=reference.log_posterior,
    nu=theta0.nu,
    untreated_cfr=theta0.untreated_cfr,
    duration=theta0.duration,
    tau_year=theta0.tau_year,
    reference_summary...,
)

comparison = DataFrame([
    reference_row,
    summary_rows...,
])

comparison.delta_loglikelihood =
    comparison.loglikelihood .-
        reference.log_likelihood

comparison.delta_logposterior =
    comparison.logposterior .-
        reference.log_posterior

CSV.write(
    joinpath(results_dir, "comparison.csv"),
    comparison,
)

println("\nFinal comparison:")
show(
    stdout, MIME"text/plain"(), comparison;
    allrows=true, allcols=true,
)
println()

println("\nParameter estimates:")
parameters = DataFrame(parameter_rows)

selected = filter(
    r -> r.parameter in (
        "nu",
        "epsilon",
        "beta",
        "psi",
        "omega",
        "untreated_cfr",
        "duration",
        "tau_year",
        "sigma_obs",
    ),
    parameters,
)

show(
    stdout, MIME"text/plain"(), selected;
    allrows=true, allcols=true,
)
println()
