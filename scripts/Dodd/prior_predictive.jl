
# scripts/Dodd/prior_predictive.jl

using TBKiribatiJulia
using Random
using CSV
using DataFrames
using CairoMakie

# ============================================================
# 1. Analysis setup
# ============================================================

include(joinpath(
    @__DIR__, "..", "common", "kiribati_setup.jl"
))

setup = setup_dodd_analysis()

model = setup.model
demography = setup.demography
observations = setup.observations
notifications = setup.notifications

start_year = setup.start_year
end_year = setup.end_year
output_years = setup.output_years
age_breaks = setup.age_breaks

output_dir = joinpath(
    setup.output_dir,
    "prior_predictive",
)

mkpath(output_dir)

# Analysis-specific configuration
n_draws = 1000
seed = 42

probabilities = [0.025, 0.5, 0.975]


# ============================================================
# 2. Sample prior
# ============================================================

rng = MersenneTwister(seed)

draws = [
    sample_prior(rng, model)
    for _ in 1:n_draws
]


# ============================================================
# 3. Simulate prior ensemble
# ============================================================

result = simulate_ensemble(
    model,
    draws,
    demography;
    start_year=start_year,
    end_year=end_year,
    output_years=output_years,
    age_breaks=age_breaks,
)

n_success = count(!isnothing, result.outputs)

println("Prior draws: ", n_draws)
println("Successful: ", n_success)
println("Failed: ", length(result.failures))

if !isempty(result.failures)

    failures = DataFrame(result.failures)

    CSV.write(
        joinpath(output_dir, "failures.csv"),
        failures,
    )

    println("\nFailure types:")
    println(combine(
        groupby(failures, :error_type),
        nrow => :count,
    ))
end

n_success > 0 ||
    error("No successful prior simulations")


# ============================================================
# 4. Individual prior predictive trajectories
# ============================================================

fig_trajectories = Figure(size=(900, 500))

ax = Axis(
    fig_trajectories[1, 1];
    xlabel="Year",
    ylabel="Annual TB notifications",
    title="Individual prior predictive trajectories",
)

for out in result.outputs

    isnothing(out) && continue

    totals = vec(sum(out.notifications; dims=1))

    lines!(
        ax,
        out.years,
        totals;
        color=(:steelblue, 0.15),
    )
end

save(
    joinpath(output_dir, "notification_trajectories.png"),
    fig_trajectories,
)


# ============================================================
# 5. Summarise predictive distributions
# ============================================================

summary = summarise_ensemble(
    result;
    probabilities=probabilities,
)


# ============================================================
# 6. Save national predictive summaries
# ============================================================

prediction_table = DataFrame(year=summary.years)

for variable in keys(summary.totals)

    values = getproperty(summary.totals, variable)

    prediction_table[!, Symbol(variable, "_lower")] =
        values[:, 1]

    prediction_table[!, Symbol(variable, "_median")] =
        values[:, 2]

    prediction_table[!, Symbol(variable, "_upper")] =
        values[:, 3]
end

CSV.write(
    joinpath(output_dir, "national_predictions.csv"),
    prediction_table,
)


# ============================================================
# 7. Plot national predictive distributions
# ============================================================

fig_national = Figure(size=(900, 850))

variables = (
    :incidence,
    :notifications,
    :mortality,
)

labels = (
    "TB incidence",
    "TB notifications",
    "TB mortality",
)

years = summary.years

for (i, variable) in enumerate(variables)

    ax = Axis(
        fig_national[i, 1];
        ylabel=labels[i],
        xlabel=i == 3 ? "Year" : "",
    )

    values = getproperty(summary.totals, variable)

    band!(
        ax,
        years,
        values[:, 1],
        values[:, 3];
        color=(:steelblue, 0.25),
    )

    lines!(
        ax,
        years,
        values[:, 2];
        color=:steelblue,
        linewidth=2,
    )
end

save(
    joinpath(output_dir, "national_predictions.png"),
    fig_national,
)


# ============================================================
# 8. Prior predictive sensitivity analysis
# ============================================================

sensitivity = sensitivity_data(
    result;
    year=2024,
)

correlations = sensitivity_correlations(sensitivity)

# Positive-valued parameters plotted on logarithmic scales.
log_parameters = [
    :beta,
    :Lambda0,
    :kappa,
    :epsilon,
    :epsilon_child,
    :nu,
    :omega,
    :duration,
]


# ------------------------------------------------------------
# 8a. Parameter pair plot
# ------------------------------------------------------------

fig_pairs = plot_parameter_pairs(
    sensitivity;
    log_parameters=log_parameters,
)

save(
    joinpath(output_dir, "parameter_pairs.png"),
    fig_pairs,
)


# ------------------------------------------------------------
# 8b. Parameter-outcome relationships
# ------------------------------------------------------------

fig_outcomes = plot_parameter_outcomes(
    sensitivity;
    log_parameters=log_parameters,
    log_outcomes=[
        :incidence,
        :notifications,
        :mortality,
    ],
)

save(
    joinpath(output_dir, "parameter_outcomes.png"),
    fig_outcomes,
)


# ============================================================
# 9. Partial rank correlation analysis
# ============================================================

prcc = sensitivity_prcc(sensitivity)

fig_prcc = plot_prcc_heatmap(prcc)

save(
    joinpath(output_dir, "parameter_prcc.png"),
    fig_prcc,
)


# ------------------------------------------------------------
# Compare marginal and partial correlations
# ------------------------------------------------------------

comparison = DataFrame(
    parameter=repeat(
        string.(prcc.parameter_names);
        outer=length(prcc.outcome_names),
    ),
    outcome=repeat(
        string.(prcc.outcome_names);
        inner=length(prcc.parameter_names),
    ),
    spearman=vec(permutedims(
        correlations.outcome_correlations
    )),
    prcc=vec(permutedims(prcc.prcc)),
)

CSV.write(
    joinpath(output_dir, "sensitivity_correlations.csv"),
    comparison,
)


# ============================================================
# 10. Inspect notification data
# ============================================================

println("\nObserved notification columns:")
println(names(notifications))

println("\nFirst five rows:")
show(stdout, MIME"text/plain"(), first(notifications, 5))
println()


# ============================================================
# 11. Completion
# ============================================================

println("\nPrior predictive analysis complete.")
println("Results saved to: ", output_dir)
