
using Test
using Statistics
using TBKiribatiJulia

module CohortWPPFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

struct PopulationModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

TB.state_layout(model::PopulationModel) = model.layout

end


@testset "Cohort demography versus WPP2024" begin
    TB = TBKiribatiJulia
    F = CohortWPPFixture

    data_dir = normpath(joinpath(@__DIR__, "..", "..", "data"))
    data = TB.load_demographic_data(data_dir)
    nages = 101

    model = F.PopulationModel(
        TB.StateLayout(1, 0, nages)
    )

    migration_age = TB.load_age_migration(
        joinpath(data_dir, "raw", "wpp2024", "migration_age.csv");
        first_year=2024,
        last_year=2050,
    )

    expected = Dict(
        :proportional => (
            population_error=-0.405416,
            age_tvd=0.905454,
        ),
        :age => (
            population_error=-0.957368,
            age_tvd=0.875173,
        ),
    )

    age_breaks = [0, 5, 15, 25, 35, 45, 55, 65, 101]

    for scheme in (:proportional, :age)
        @testset "$scheme migration" begin
            demo = TB.CohortDemography(
                data;
                nages=nages,
                female_fraction=0.5,
                migration_scheme=scheme,
                age_migration=scheme == :age ?
                              migration_age : nothing,
                migration_first_year=2024,
            )

            # Initialise from WPP in 2023.
            u = TB.get_population(data, 2023; nages)

            @test length(u) == nages

            initial_population = sum(u)

            # Independent annual projection.
            for year in 2023:2049
                TB.apply_demography!(
                    u, model, demo, year
                )

                @test all(isfinite, u)
                @test all(u .>= 0)
            end

            W = TB.get_population(
                data, 2050; nages
            )

            population_error =
                100 * (sum(u) / sum(W) - 1)

            age_tvd = 50 * sum(abs.(
                u ./ sum(u) .- W ./ sum(W)
            ))

            age_errors = Float64[]

            for g in 1:8
                lo = age_breaks[g]
                hi = age_breaks[g+1]

                inds = (lo+1):hi

                push!(
                    age_errors,
                    100 * (
                        sum(@view u[inds]) /
                            sum(@view W[inds]) - 1
                    ),
                )
            end

            println("\nMigration: ", scheme)
            println("Initial population: ", initial_population)
            println("2050 projected population: ", sum(u))
            println("2050 WPP population: ", sum(W))
            println("Population error (%): ", population_error)
            println("Age TVD (%): ", age_tvd)
            println("Maximum age-group error (%): ",
                maximum(abs.(age_errors)))

            # Compare with the independently implemented
            # midyear-fertility projection.
            @test population_error ≈
                expected[scheme].population_error atol=0.05

            @test age_tvd ≈
                expected[scheme].age_tvd atol=0.05

            @test maximum(abs.(age_errors)) < 5.0
        end
    end
end
