
using TBKiribatiJulia
using DataFrames, CSV, Random, Optim
using Distributions: Normal

# --------------------------------------------------
# Setup
# --------------------------------------------------

reference = fitted_models["F_nu_only"]
problem = problem100
theta0 = reference.parameters

results_dir_detection = normpath(joinpath(
    @__DIR__, "..", "..", "results",
    "DoddExtended", "detection_experiments",
))
mkpath(results_dir_detection)

# Keep nu free; fix kappa, duration and untreated CFR.
free_base = (
    :beta, :psi, :Lambda0, :epsilon, :nu, :omega,
    :detection_logitK, :extrapulmonary_detection,
    :sigma_obs, :tau_year,
)

function detection_fixed(problem, theta, free)
    names = Tuple(
        k for k in keys(problem.prior) if !(k in free)
    )
    return NamedTuple{names}(
        Tuple(getproperty(theta, k) for k in names)
    )
end

function detection_summary(problem, theta; year=2024)
    out = predict(
        problem, theta;
        variables=(
            :notifications, :incidence, :mortality,
            :untreated_mortality, :treatment_mortality,
        ),
    )

    j = findfirst(==(year), out.years)
    isnothing(j) && error("Missing prediction year")

    total(name) = sum(getproperty(out, name)[:, j])

    K = 1 / (1 + exp(-theta.detection_logitK))

    detection_at(year) = begin
        z = theta.detection_logitK +
            theta.detection_trend * (year - 1970)
        0.9 / (1 + exp(-z))
    end

    return (
        K=K,
        trend=theta.detection_trend,
        detection_1980=detection_at(1980),
        detection_2000=detection_at(2000),
        detection_2024=detection_at(2024),
        notifications=total(:notifications),
        incidence=total(:incidence),
        mortality=total(:mortality),
        untreated_deaths=total(:untreated_mortality),
        treatment_deaths=total(:treatment_mortality),
        tau_year=theta.tau_year,
    )
end

results = NamedTuple[]

function record_detection_fit!(results, name, trial)
    values = detection_summary(
        trial.problem, trial.parameters
    )

    push!(results, (
        experiment=name,
        converged=trial.converged,
        iterations=Optim.iterations(trial.optim_result),
        loglikelihood=trial.log_likelihood,
        logposterior=trial.log_posterior,
        values...,
    ))

    CSV.write(
        joinpath(results_dir_detection, "comparison.csv"),
        DataFrame(results),
    )

    println("\n", name)
    println("Converged: ", trial.converged)
    println("Log likelihood: ", trial.log_likelihood)
    println("2024 outcomes: ", values)
end

# --------------------------------------------------
# Experiment K: Higher detection starting values
# --------------------------------------------------

fixed_K = detection_fixed(
    problem, theta0, free_base
)

# Trend is explicitly fixed at zero.
fixed_K = merge(fixed_K, (detection_trend=0.0,))

for (i, K0) in enumerate((0.4095748, 0.65, 0.85))
    name = "K_$(i)"

    initial = merge(
        theta0,
        (
            detection_logitK=log(K0 / (1-K0)),
            detection_trend=0.0,
        ),
    )

    trial = fit_map(
        problem;
        fixed=fixed_K,
        initial=initial,
        optimizer=:nelder_mead,
        nstarts=1,
        iterations=3000,
        rng=MersenneTwister(300+i),
    )

    fitted_models[name] = trial
    record_detection_fit!(results, name, trial)
end

# --------------------------------------------------
# Experiment L: Signed detection trend
# --------------------------------------------------

# Replace only the trend prior.
# Reuse the prepared simulation and existing observations.

signed_prior = merge(
    problem.prior,
    (detection_trend=Normal(0.0, 0.02),),
)

problem_signed = CalibrationProblem(
    problem.model,
    problem.demography,
    problem.observations,
    problem.observation_model,
    signed_prior,
    problem.simulation,
    problem.age_breaks,
    problem.start_year,
    problem.end_year,
    problem.treatment_deaths,
    problem.recurrence,
    problem.site,
)

free_L = (free_base..., :detection_trend)

fixed_L = detection_fixed(
    problem_signed, theta0, free_L
)

# Use several initial trends to check for local optima.
for (i, r0) in enumerate((-0.02, 0.0, 0.02))
    name = "L_$(i)"

    initial = merge(
        theta0,
        (detection_trend=r0,),
    )

    trial = fit_map(
        problem_signed;
        fixed=fixed_L,
        initial=initial,
        optimizer=:nelder_mead,
        nstarts=1,
        iterations=3000,
        rng=MersenneTwister(400+i),
    )

    fitted_models[name] = trial
    record_detection_fit!(results, name, trial)
end

# --------------------------------------------------
# Compare results
# --------------------------------------------------

comparison_detection = DataFrame(results)

comparison_detection.delta_loglikelihood =
    comparison_detection.loglikelihood .-
        reference.log_likelihood

println("\nDETECTION EXPERIMENT COMPARISON")
show(
    stdout,
    MIME"text/plain"(),
    comparison_detection;
    allrows=true,
    allcols=true,
)
println()

CSV.write(
    joinpath(results_dir_detection, "comparison.csv"),
    comparison_detection,
)

# --------------------------------------------------
# Presentation-weighted CFR diagnostic
# --------------------------------------------------

out_ref = predict(
    problem,
    reference.parameters;
    variables=(:incidence, :untreated_mortality,
        :treatment_mortality),
)

j = findfirst(==(2024), out_ref.years)

# Actual model presentation proportions by age group,
# weighted by the fitted age-specific incidence.
breaks = problem.age_breaks
P = problem.model.presentation

inc_by_age = vec(out_ref.incidence[:, j])

site_weights = zeros(3)

for g in eachindex(inc_by_age)
    age_index = breaks[g] + 1

    # Presentation probabilities are constant within
    # each calibration age group.
    site_weights .+=
        inc_by_age[g] .* P[age_index, :]
end

site_weights ./= sum(site_weights)

site_cfr = [0.70, 0.20, 0.20]
weighted_cfr = sum(site_weights .* site_cfr)

untreated = sum(out_ref.untreated_mortality[:, j])
treatment = sum(out_ref.treatment_mortality[:, j])

# Illustrative rescaling, NOT a dynamic counterfactual.
adjusted_deaths =
    untreated * weighted_cfr / theta0.untreated_cfr +
        treatment

println("\nPRESENTATION-WEIGHTED CFR")
println("-------------------------")
println("2024 incidence shares (SP, SN, EP): ", site_weights)
println("Weighted untreated CFR: ", weighted_cfr)
println("Original TB deaths: ", untreated + treatment)
println("Illustrative adjusted TB deaths: ", adjusted_deaths)
println("Note: adjusted deaths are not simulated.")
