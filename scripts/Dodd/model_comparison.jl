
# scripts/Dodd/model_comparison.jl

using TBKiribatiJulia
using Random
using DataFrames
using CSV
using CairoMakie

include(joinpath(
    @__DIR__, "..", "common", "observation_models.jl"
))

setup = setup_dodd_analysis(
    start_year=1970,
    end_year=2025,
    output_years=2013:2024,
    use_historical_notifications=false,
)

obs_independent = kiribati_observation_model(
    annual_effects=false,
)

obs_annual = kiribati_observation_model(
    annual_effects=true,
)

# Independent NB2 observation model.
problem_independent = CalibrationProblem(
    setup.model,
    setup.demography,
    setup.observations;
    start_year=setup.start_year,
    end_year=setup.end_year,
    age_breaks=setup.age_breaks,
    observation_model=obs_independent,
)

# NB2 with shared annual random effects.
problem_annual = CalibrationProblem(
    setup.model,
    setup.demography,
    setup.observations;
    start_year=setup.start_year,
    end_year=setup.end_year,
    age_breaks=setup.age_breaks,
    observation_model=obs_annual,
)

# Maximum-likelihood model comparison.
fits = (
    M1=fit_mle(
        problem_independent;
        fixed=(detection_trend=0.0,),
        rng=MersenneTwister(101),
        nstarts=3,
        iterations=5000,
    ),
    M2=fit_mle(
        problem_annual;
        fixed=(detection_trend=0.0,),
        rng=MersenneTwister(102),
        nstarts=3,
        iterations=5000,
    ),
    M3=fit_mle(
        problem_independent;
        rng=MersenneTwister(103),
        nstarts=3,
        iterations=5000,
    ),
    M4=fit_mle(
        problem_annual;
        rng=MersenneTwister(104),
        nstarts=3,
        iterations=5000,
    ),
)

comparison = DataFrame(compare_calibrations(fits))

println("\nMaximum-likelihood model comparison:")
display(comparison)

lrt = (
    annual_without_trend=likelihood_ratio(
        fits.M1, fits.M2
    ),
    annual_with_trend=likelihood_ratio(
        fits.M3, fits.M4
    ),
    trend_without_annual=likelihood_ratio(
        fits.M1, fits.M3
    ),
    trend_with_annual=likelihood_ratio(
        fits.M2, fits.M4
    ),
)

println("\nLikelihood-ratio comparisons:")
display(lrt)

# Save comparative results.
output_dir = joinpath(
    setup.output_dir, "model_comparison"
)

mkpath(output_dir)

CSV.write(
    joinpath(output_dir, "model_comparison.csv"),
    comparison,
)

# Initial profiles of the full model.
profile_parameters = [
    :beta,
    :nu,
    :Lambda0,
    :detection_logitK,
    :detection_trend,
    :duration,
    :sigma_obs,
    :tau_year,
]

profiles = profile_likelihoods(
    fits.M4;
    parameters=profile_parameters,
    nstarts=2,
    iterations=3000,
    rng=MersenneTwister(200),
)

for (name, profile) in profiles
    fig = plot_profile_likelihood(profile)

    save(
        joinpath(
            output_dir,
            "profile_$(name).png",
        ),
        fig,
    )

    CSV.write(
        joinpath(
            output_dir,
            "profile_$(name).csv",
        ),
        DataFrame([
            (
                value=p.value,
                log_likelihood=p.log_likelihood,
                deviance=p.deviance,
                converged=p.converged,
            )
            for p in profile.points
        ]),
    )
end
