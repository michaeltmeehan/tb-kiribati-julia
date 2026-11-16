
using TBKiribatiJulia
using Distributions: Normal, truncated
using Statistics: median
using DataFrames
using CSV
using Test

include(joinpath(
    @__DIR__, "..", "common", "kiribati_setup.jl"
))

setup = setup_dodd_analysis(
    start_year=1970,
    end_year=2026,
    output_years=2013:2025,
    use_historical_notifications=true,
)

obsmodel = AnnualRandomEffectNB2Observation(
    truncated(Normal(0, 0.3), 0, Inf),
    truncated(Normal(0, 0.2), 0, Inf);
    quadrature_order=30,
)

# Treatment-death observations, initially 2012–2023.
deaths = TreatmentDeathObservation(
    setup.surveillance.treatment_outcomes;
    include_2024=false,
)

# Original notification-only problem.
p0 = CalibrationProblem(
    setup.model,
    setup.demography,
    setup.observations;
    start_year=setup.start_year,
    end_year=setup.end_year,
    age_breaks=setup.age_breaks,
    observation_model=obsmodel,
)

# Extended problem including treatment deaths.
p1 = CalibrationProblem(
    setup.model,
    setup.demography,
    setup.observations;
    start_year=setup.start_year,
    end_year=setup.end_year,
    age_breaks=setup.age_breaks,
    observation_model=obsmodel,
    treatment_deaths=deaths,
)

theta = map(median, p1.prior)

@test haskey(theta, :treatment_cfr)

# Only simulate once.
@time output = predict(p0, theta)

ll0 = calibration_loglikelihood(
    p0, theta; output=output
)

ll_deaths = treatment_death_loglikelihood(
    deaths, theta
)

ll1 = calibration_loglikelihood(
    p1, theta; output=output
)

@testset "Treatment-death likelihood" begin

    @test length(deaths.rows) == 12

    @test 0 <= treatment_death_probability(
            deaths, theta
        ) <= 1

    @test isfinite(ll0)
    @test isfinite(ll_deaths)
    @test isfinite(ll1)

    @test isapprox(
        ll1,
        ll0 + ll_deaths;
        atol=1e-8,
    )

end

println("Notification log likelihood: ", ll0)
println("Death log likelihood:        ", ll_deaths)
println("Combined log likelihood:     ", ll1)

death_diagnostics = treatment_death_fit_rows(
    deaths, theta
)

show(stdout, MIME"text/plain"(),
    DataFrame(death_diagnostics))
println()
