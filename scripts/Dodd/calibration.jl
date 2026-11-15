
# scripts/Dodd/calibration.jl

using TBKiribatiJulia
using Random
using Distributions: Normal, truncated

include(joinpath(
    @__DIR__,
    "..",
    "common",
    "kiribati_setup.jl",
))

# ------------------------------------------------------------
# Shared Kiribati setup
# ------------------------------------------------------------

setup = setup_dodd_analysis()

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
    iterations=3000,
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
# Summary
# ------------------------------------------------------------

println("MAP converged: ", fit.converged)
println("Log likelihood: ", fit.log_likelihood)
println("Log posterior: ", fit.log_posterior)

println("sigma_obs: ", fit.parameters.sigma_obs)
println("tau_year: ", fit.parameters.tau_year)

println("Results: ", output_dir)
