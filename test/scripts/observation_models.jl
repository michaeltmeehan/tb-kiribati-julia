using Test
using TBKiribatiJulia
using Distributions: median

include(joinpath(
    @__DIR__, "..", "..",
    "scripts", "common", "observation_models.jl",
))

@testset "Kiribati observation-model defaults" begin
    independent = kiribati_observation_model(
        annual_effects=false,
    )

    annual = kiribati_observation_model()

    @test independent isa NB2Observation
    @test annual isa AnnualRandomEffectNB2Observation

    @test length(annual.nodes) == 30
    @test length(annual.weights) == 30

    @test keys(observation_priors(independent)) ==
        (:sigma_obs,)

    @test keys(observation_priors(annual)) ==
        (:sigma_obs, :tau_year)

    @test observation_priors(annual).sigma_obs ==
        observation_priors(independent).sigma_obs

    @test_throws ArgumentError kiribati_observation_model(
        sigma_sd=0.0,
    )
end