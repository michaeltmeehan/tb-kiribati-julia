
using TBKiribatiJulia
using DataFrames, CSV, Random
using Optim

# --------------------------------------------------
# 1. Additional parameter-release experiments
# --------------------------------------------------

additional_experiments = [
    ("F_nu_only", (:nu,)),
    ("G_kappa_only", (:kappa,)),
]

new_results = NamedTuple[]

for (i, (name, released)) in enumerate(additional_experiments)
    println("\n", "="^60)
    println("Experiment: ", name)
    println("Released: ", released)

    free_names = (base_free..., released...)

    fixed = fixed_parameters(
        problem100, free_names, theta0
    )

    @assert all(k -> !haskey(fixed, k), free_names)

    trial = fit_map(
        problem100;
        fixed=fixed,
        initial=theta0,
        optimizer=:nelder_mead,
        nstarts=1,
        iterations=3000,
        rng=MersenneTwister(100 + i),
    )

    fitted_models[name] = trial

    prediction = annual_summary(
        problem100, trial.parameters
    )

    push!(new_results, (
        experiment=name,
        converged=trial.converged,
        iterations=Optim.iterations(trial.optim_result),
        loglikelihood=trial.log_likelihood,
        logposterior=trial.log_posterior,
        nu=trial.parameters.nu,
        kappa=trial.parameters.kappa,
        epsilon=trial.parameters.epsilon,
        tau_year=trial.parameters.tau_year,
        prediction...,
    ))

    println("Converged: ", trial.converged)
    println("Log likelihood: ", trial.log_likelihood)
    println("Log posterior: ", trial.log_posterior)
    println("2024 predictions: ", prediction)
end

println("\nAdditional experiment results:")
display(DataFrame(new_results))

# --------------------------------------------------
# 2. Mortality decomposition
# --------------------------------------------------

function mortality_decomposition(
    problem,
    theta;
    years=(2023, 2024, 2025),
)
    out = predict(
        problem, theta;
        variables=(
            :notifications,
            :incidence,
            :mortality,
            :untreated_mortality,
            :treatment_mortality,
            :self_cure,
            :treatment_completion,
            :fast_progression,
            :slow_progression,
            :relapse,
        ),
    )

    rows = NamedTuple[]

    for year in years
        j = findfirst(==(year), out.years)
        isnothing(j) && error("Missing prediction year $year")
        total(variable) =
            sum(getproperty(out, variable)[:, j])

        untreated = total(:untreated_mortality)
        treatment = total(:treatment_mortality)
        deaths = total(:mortality)

        @assert isapprox(
            deaths, untreated + treatment;
            rtol=1e-6, atol=1e-6,
        )

        incidence = total(:incidence)

        @assert isapprox(
            incidence,
            total(:fast_progression) +
                total(:slow_progression) +
                total(:relapse);
            rtol=1e-6, atol=1e-6,
        )

        push!(rows, (
            year=year,
            notifications=total(:notifications),
            incidence=incidence,
            deaths=deaths,
            untreated_deaths=untreated,
            treatment_deaths=treatment,
            untreated_death_fraction=deaths > 0 ?
                                     untreated / deaths : NaN,
            deaths_per_incident_case=incidence > 0 ?
                                     deaths / incidence : NaN,
            self_cure=total(:self_cure),
            treatment_completion=total(:treatment_completion),
            fast_progression=total(:fast_progression),
            slow_progression=total(:slow_progression),
            relapse=total(:relapse),
        ))
    end

    return DataFrame(rows)
end

# --------------------------------------------------
# 3. Compare baseline, D, F and G
# --------------------------------------------------

models_to_compare = [
    "A_baseline",
    "D_progression",
    "F_nu_only",
    "G_kappa_only",
]

mortality_rows = DataFrame[]
comparison_rows = NamedTuple[]

for name in models_to_compare
    trial = fitted_models[name]

    mortality = mortality_decomposition(
        problem100, trial.parameters
    )

    mortality.experiment .= name
    push!(mortality_rows, mortality)

    r = only(filter(row -> row.year == 2024, mortality))

    push!(comparison_rows, (
        experiment=name,
        converged=trial.converged,
        loglikelihood=trial.log_likelihood,
        logposterior=trial.log_posterior,
        nu=trial.parameters.nu,
        kappa=trial.parameters.kappa,
        epsilon=trial.parameters.epsilon,
        beta=trial.parameters.beta,
        psi=trial.parameters.psi,
        tau_year=trial.parameters.tau_year,
        notifications=r.notifications,
        incidence=r.incidence,
        mortality=r.deaths,
        untreated_deaths=r.untreated_deaths,
        treatment_deaths=r.treatment_deaths,
        untreated_death_fraction=r.untreated_death_fraction,
        deaths_per_incident_case=r.deaths_per_incident_case,
    ))
end

comparison = DataFrame(comparison_rows)

comparison.delta_loglikelihood =
    comparison.loglikelihood .-
        comparison.loglikelihood[1]

comparison.delta_logposterior =
    comparison.logposterior .-
        comparison.logposterior[1]

mortality_results = vcat(mortality_rows...)

# --------------------------------------------------
# 4. Save and display
# --------------------------------------------------

CSV.write(
    joinpath(results_dir, "progression_comparison.csv"),
    comparison,
)

CSV.write(
    joinpath(results_dir, "mortality_decomposition.csv"),
    mortality_results,
)

println("\nProgression comparison:")
show(
    stdout, MIME"text/plain"(), comparison;
    allrows=true, allcols=true,
)
println()

println("\n2024 mortality and progression decomposition:")
show(
    stdout,
    MIME"text/plain"(),
    filter(r -> r.year == 2024, mortality_results);
    allrows=true, allcols=true,
)
println()
