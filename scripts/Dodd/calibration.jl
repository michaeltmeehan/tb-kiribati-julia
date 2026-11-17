
# scripts/Dodd/calibration.jl

using TBKiribatiJulia
using Random
using Distributions: Normal, truncated
using CSV
using DataFrames
using CairoMakie

include(joinpath(
    @__DIR__,
    "..",
    "common",
    "kiribati_setup.jl",
))

# ------------------------------------------------------------
# Shared Kiribati setup
# ------------------------------------------------------------

setup = setup_dodd_analysis(
    start_year=1970,
    end_year=2026,
    output_years=2013:2025,
    use_historical_notifications=true,
    national_start_year=1980,
)

# ------------------------------------------------------------
# Observation-model priors
# ------------------------------------------------------------

sigma_prior = truncated(
    Normal(0.0, 0.3),
    0.0,
    Inf,
)

tau_prior = truncated(
    Normal(0.0, 0.2),
    0.0,
    Inf,
)

# ------------------------------------------------------------
# Calibration problem
# ------------------------------------------------------------

problem = CalibrationProblem(
    setup.model,
    setup.demography,
    setup.observations;
    start_year=setup.start_year,
    end_year=setup.end_year,
    treatment_deaths=TreatmentDeathObservation(
        setup.surveillance.treatment_outcomes;
        include_2024=false,
    ),
    age_breaks=setup.age_breaks,
    observation_model=AnnualRandomEffectNB2Observation(
        sigma_prior,
        tau_prior;
        quadrature_order=30,
    ),
)

# ------------------------------------------------------------
# MAP calibration
# ------------------------------------------------------------

fit = fit_map(
    problem;
    rng=MersenneTwister(42),
    nstarts=3,
    iterations=10_000,
)

# ------------------------------------------------------------
# Diagnostics and plots
# ------------------------------------------------------------

diagnostics = calibration_diagnostics(fit)

fig_national = plot_calibration_national(
    fit;
    diagnostics=diagnostics,
)

fig_age = plot_calibration_by_age(
    fit;
    diagnostics=diagnostics,
)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

output_dir = save_calibration(
    fit,
    joinpath(
        setup.output_dir,
        "calibration_annual_effect",
    );
    diagnostics=diagnostics,
    figures=(
        national=fig_national,
        age=fig_age,
    ),
)


# ------------------------------------------------------------
# Quadrature convergence at the MAP estimate
# ------------------------------------------------------------

quadrature_check = quadrature_convergence(
    fit;
    orders=(30, 50, 70, 100),
)

println("\nQuadrature convergence:")
show(stdout, MIME"text/plain"(), DataFrame(quadrature_check))
println()

CSV.write(
    joinpath(output_dir, "quadrature_convergence.csv"),
    DataFrame(quadrature_check),
)

# ------------------------------------------------------------
# Conditional posterior annual notification effects
# ------------------------------------------------------------

annual_effects = posterior_annual_effects(
    fit;
    quadrature_order=100,
    level=0.95,
)

println("\nConditional annual effects:")
show(stdout, MIME"text/plain"(), DataFrame(annual_effects))
println()

CSV.write(
    joinpath(output_dir, "annual_effects.csv"),
    DataFrame(annual_effects),
)

fig_annual_effects = plot_annual_effects(annual_effects)

save(
    joinpath(output_dir, "annual_effects.png"),
    fig_annual_effects,
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

println("MAP converged: ", fit.converged)
println("Log likelihood: ", fit.log_likelihood)
println("Log posterior: ", fit.log_posterior)

println("sigma_obs: ", fit.parameters.sigma_obs)
println("tau_year: ", fit.parameters.tau_year)

println("Results: ", output_dir)



# ------------------------------------------------------------
# Maximum likelihood calibration
# ------------------------------------------------------------

println("\nRunning maximum likelihood calibration...")

treatment_cfr_mle = 304 / 4928
duration_mle = 3.

mle = fit_mle(
    problem;
    fixed=(treatment_cfr=treatment_cfr_mle, duration=duration_mle),
    initial=fit.parameters,  # Start from the MAP estimate
    rng=MersenneTwister(43),
    nstarts=1,
    iterations=3_000,
)

println("\nMLE converged: ", mle.converged)
println("MAP log likelihood: ", fit.log_likelihood)
println("MLE log likelihood: ", mle.log_likelihood)
println(
    "Improvement: ",
    mle.log_likelihood - fit.log_likelihood,
)

if mle.log_likelihood < fit.log_likelihood - 1e-6
    @warn "MLE is worse than MAP: further optimization is needed."
end


# ------------------------------------------------------------
# MLE diagnostics and reporting
# ------------------------------------------------------------

mle_diagnostics = calibration_diagnostics(mle)

mle_fig_national = plot_calibration_national(
    mle;
    diagnostics=mle_diagnostics,
)

mle_fig_age = plot_calibration_by_age(
    mle;
    diagnostics=mle_diagnostics,
)

mle_dir = save_calibration(
    mle,
    joinpath(setup.output_dir, "calibration_mle");
    diagnostics=mle_diagnostics,
    figures=(
        national=mle_fig_national,
        age=mle_fig_age,
    ),
)


# ------------------------------------------------------------
# Compare MAP and MLE parameter estimates
# ------------------------------------------------------------

parameter_names = collect(keys(fit.parameters))

parameter_comparison = DataFrame(
    parameter=String.(parameter_names),
    map=[
        Float64(getproperty(fit.parameters, name))
        for name in parameter_names
    ],
    mle=[
        Float64(getproperty(mle.parameters, name))
        for name in parameter_names
    ],
)

parameter_comparison.difference =
    parameter_comparison.mle .- parameter_comparison.map

println("\nMAP versus MLE:")
show(
    stdout,
    MIME"text/plain"(),
    parameter_comparison;
    allrows=true,
)
println()

CSV.write(
    joinpath(mle_dir, "map_vs_mle.csv"),
    parameter_comparison,
)


# ------------------------------------------------------------
# Quadrature convergence at the MLE
# ------------------------------------------------------------

mle_quadrature = quadrature_convergence(
    mle;
    orders=(30, 50, 70, 100),
)

CSV.write(
    joinpath(mle_dir, "quadrature_convergence.csv"),
    DataFrame(mle_quadrature),
)

println("\nMLE results saved to: ", mle_dir)
