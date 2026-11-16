
# scripts/Dodd/check_historical_likelihood.jl

using TBKiribatiJulia
using Distributions: Normal, truncated
using Statistics: median
using DataFrames
using CSV
using Test
using CairoMakie

include(joinpath(
    @__DIR__, "..", "common", "kiribati_setup.jl"
))

setup = setup_dodd_analysis(
    start_year=1970,
    end_year=2026,
    output_years=2013:2025,
    use_historical_notifications=true,
    national_start_year=1980,
)

observation_model = AnnualRandomEffectNB2Observation(
    truncated(Normal(0, 0.3), 0, Inf),
    truncated(Normal(0, 0.2), 0, Inf);
    quadrature_order=30,
)

problem = CalibrationProblem(
    setup.model,
    setup.demography,
    setup.observations;
    start_year=setup.start_year,
    end_year=setup.end_year,
    age_breaks=setup.age_breaks,
    observation_model=observation_model,
)

# A deterministic parameter point for checking the likelihood.
theta = map(median, problem.prior)

println("Running one simulation...")
@time output = predict(problem, theta)

println("Evaluating likelihood...")
@time ll = calibration_loglikelihood(
    problem, theta; output=output
)
lp = calibration_logprior(problem, theta)

rows = notification_fit_rows(problem, output)

@testset "Historical likelihood" begin
    @test length(problem.observations) == 45
    @test length(rows) == 135

    @test all(isfinite, output.notifications)
    @test all(>=(0), output.notifications)

    @test all(isfinite(r.expected) && r.expected >= 0
              for r in rows)

    @test isfinite(ll)
    @test isfinite(lp)

    @test ll == calibration_loglikelihood(
        problem, theta; output=output
    )
end

println("Log likelihood: ", ll)
println("Log prior:      ", lp)
println("Log posterior:  ", ll + lp)
