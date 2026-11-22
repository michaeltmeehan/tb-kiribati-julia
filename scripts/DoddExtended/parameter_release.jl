
using TBKiribatiJulia
using Random, CSV, DataFrames
using Distributions: Normal, truncated
using Optim

# --------------------------------------------------
# Configuration
# --------------------------------------------------

const TB = TBKiribatiJulia

baseline_fit = fit
original_problem = baseline_fit.problem
theta0 = baseline_fit.parameters

results_dir = joinpath(
    @__DIR__, "..", "..", "results",
    "DoddExtended", "parameter_release",
)
mkpath(results_dir)

# --------------------------------------------------
# Reuse prepared simulation with improved quadrature
# --------------------------------------------------

old_obs = original_problem.observation_model

observation_model = AnnualRandomEffectNB2Observation(
    old_obs.sigma_prior,
    old_obs.tau_prior;
    quadrature_order=100,
)

# Construct using the existing prepared simulation.
# This avoids repeating expensive initialisation.
@assert fieldnames(typeof(original_problem)) == (
    :model, :demography, :observations,
    :observation_model, :prior, :simulation,
    :age_breaks, :start_year, :end_year,
    :treatment_deaths, :recurrence, :site,
)

problem100 = CalibrationProblem(
    original_problem.model,
    original_problem.demography,
    original_problem.observations,
    observation_model,
    original_problem.prior,
    original_problem.simulation,
    original_problem.age_breaks,
    original_problem.start_year,
    original_problem.end_year,
    original_problem.treatment_deaths,
    original_problem.recurrence,
    original_problem.site,
)

# --------------------------------------------------
# Parameter configurations
# --------------------------------------------------

base_free = (
    :beta,
    :psi,
    :Lambda0,
    :epsilon,
    :omega,
    :detection_logitK,
    :extrapulmonary_detection,
    :sigma_obs,
    :tau_year,
)

experiments = [
    ("A_baseline", ()),
    ("B_mortality", (:untreated_cfr,)),
    ("C_duration", (:duration,)),
    ("D_progression", (:kappa, :nu)),
    ("E_combined", (
        :untreated_cfr, :duration, :kappa, :nu,
    )),
]

function fixed_parameters(problem, free_names, theta)
    names = Tuple(
        k for k in keys(problem.prior)
              if !(k in free_names)
    )

    return NamedTuple{names}(
        Tuple(getproperty(theta, k) for k in names)
    )
end

function annual_summary(problem, theta; year=2024)
    out = predict(
        problem, theta;
        variables=(
            :incidence,
            :mortality,
            :notifications,
            :recurrent_notifications,
            :pulmonary_notifications,
        ),
    )

    j = findfirst(==(year), out.years)
    isnothing(j) && error("Prediction year missing")

    notifications = sum(out.notifications[:, j])
    incidence = sum(out.incidence[:, j])
    mortality = sum(out.mortality[:, j])
    recurrent = sum(out.recurrent_notifications[:, j])
    pulmonary = sum(out.pulmonary_notifications[:, j])

    return (
        notifications=notifications,
        incidence=incidence,
        mortality=mortality,
        recurrent_fraction=recurrent / notifications,
        pulmonary_fraction=pulmonary / notifications,
    )
end

# --------------------------------------------------
# Run experiments
# --------------------------------------------------

summary_rows = NamedTuple[]
parameter_rows = NamedTuple[]
fitted_models = Dict{String,Any}()

for (i, (name, released)) in enumerate(experiments)

    println("\n", "="^60)
    println("Experiment: ", name)
    println("Released: ", released)
    println("="^60)

    free_names = (base_free..., released...)

    fixed = fixed_parameters(
        problem100, free_names, theta0
    )

    @assert all(k -> !haskey(fixed, k), free_names)

    # Start all experiments at the same parameter vector.
    initial = theta0

    trial = fit_map(
        problem100;
        fixed=fixed,
        initial=initial,
        optimizer=:nelder_mead,
        nstarts=1,
        iterations=2000,
        rng=MersenneTwister(42 + i),
    )

    fitted_models[name] = trial

    prediction = annual_summary(
        problem100, trial.parameters
    )

    push!(summary_rows, (
        experiment=name,
        n_free=length(free_names),
        converged=trial.converged,
        iterations=Optim.iterations(trial.optim_result),
        loglikelihood=trial.log_likelihood,
        logprior=trial.log_prior,
        logposterior=trial.log_posterior,
        prediction...,
    ))

    for k in keys(problem100.prior)
        push!(parameter_rows, (
            experiment=name,
            parameter=String(k),
            initial=Float64(getproperty(theta0, k)),
            fitted=Float64(getproperty(trial.parameters, k)),
            free=(k in free_names),
        ))
    end

    # Save after every experiment.
    CSV.write(
        joinpath(results_dir, "summary.csv"),
        DataFrame(summary_rows),
    )

    CSV.write(
        joinpath(results_dir, "parameters.csv"),
        DataFrame(parameter_rows),
    )

    println("\nConverged: ", trial.converged)
    println("Log likelihood: ", trial.log_likelihood)
    println("Log posterior: ", trial.log_posterior)
    println("2024 predictions: ", prediction)
end

# --------------------------------------------------
# Comparison
# --------------------------------------------------

comparison = DataFrame(summary_rows)
comparison.delta_loglikelihood =
    comparison.loglikelihood .-
        comparison.loglikelihood[1]

comparison.delta_logposterior =
    comparison.logposterior .-
        comparison.logposterior[1]

println("\nFinal comparison:")
show(
    stdout,
    MIME"text/plain"(),
    comparison;
    allrows=true,
    allcols=true,
)
println()

CSV.write(
    joinpath(results_dir, "comparison.csv"),
    comparison,
)

parameters = DataFrame(parameter_rows)

println("\nReleased parameter estimates:")

selected = filter(
    r -> r.parameter in (
        "untreated_cfr",
        "duration",
        "kappa",
        "nu",
        "beta",
        "psi",
        "epsilon",
        "omega",
        "tau_year",
    ),
    parameters,
)

show(
    stdout,
    MIME"text/plain"(),
    selected;
    allrows=true,
    allcols=true,
)
println()
