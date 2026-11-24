using Test
using TBKiribatiJulia

@testset "Growth-only demography" begin
    data = DemographicData(
        repeat(reshape(collect(1.0:100.0), :, 1), 1, 3) .*
            [1.0 1.1 1.2],
        zeros(40, 3),
        zeros(100, 3),
        zeros(3),
        2025,
        2027,
        10,
        0.485,
    )

    reference_year = 2025

    fixed = EquilibriumDemography(data, reference_year)
    growth = GrowthOnlyDemography(data, reference_year)
    projected = ProjectedDemography(data)

    N0 = get_population(projected, reference_year)

    for year in 2025:2027
        N_fixed = get_population(fixed, year)
        N_growth = get_population(growth, year)
        N_projected = get_population(projected, year)

        @test N_fixed ≈ N0
        @test sum(N_growth) ≈ sum(N_projected)
        @test N_growth ./ sum(N_growth) ≈ N0 ./ sum(N0)
        @test demographic_mortality_year(
            growth, year
        ) == reference_year
    end

    @test target_population(growth, 2025) ≈
        get_population(growth, 2026)
end