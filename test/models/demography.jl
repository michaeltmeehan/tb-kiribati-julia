
using Test
using ForwardDiff
using TBKiribatiJulia

module DemographyFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

struct TestModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

TB.state_layout(model::TestModel) = model.layout

function make_data(;
    mortality=zeros(3, 2),
    population=[
        0.10 0.15
        0.20 0.25
        0.30 0.35
    ],
)
    return TB.DemographicData(
        Matrix{Float64}(population),
        zeros(1, 2),
        Matrix{Float64}(mortality),
        zeros(2),
        2023,
        2024,
        10,
        0.485,
    )
end

model() = TestModel(TB.StateLayout(2, 1, 3))

function state(values)
    # Rows are ages; columns are S, I, counter.
    return vec(permutedims(values))
end

end


@testset "Demographic operators" begin
    TB = TBKiribatiJulia
    F = DemographyFixture

    @testset "Mortality and ageing" begin
        mortality = zeros(3, 2)
        mortality[1, 1] = log(2)

        data = F.make_data(; mortality)
        layout = TB.state_layout(F.model())

        # Age 0: 100 S
        # Age 1: 200 I
        # Age 2+: 300 I
        u = F.state([
            100.0 0.0 4.0
            0.0 200.0 5.0
            0.0 300.0 6.0
        ])

        updated = zeros(3, 2)

        TB.age_and_survive!(
            updated, u, layout, data, 2023
        )

        expected = [
            0.0 0.0
            50.0 0.0
            0.0 500.0
        ]

        @test updated ≈ expected

        # Age 0 survives at 50%, then ages to 1.
        # Ages 1 and 2 combine in the open-ended age 2+.
        @test sum(updated) ≈ 550.0

        # Source state and counters are unchanged.
        @test u == F.state([
            100.0 0.0 4.0
            0.0 200.0 5.0
            0.0 300.0 6.0
        ])
    end

    @testset "Population reconciliation" begin
        updated = [
            0.0 0.0
            50.0 0.0
            0.0 500.0
        ]

        TB.reconcile_population!(
            updated, [100.0, 200.0, 300.0]
        )

        @test updated ≈ [
            100.0 0.0
            200.0 0.0
            0.0 300.0
        ]

        @test vec(sum(updated; dims=2)) ≈
            [100.0, 200.0, 300.0]

        # Preserve within-age compartment proportions.
        mixed = reshape([20.0, 80.0], 1, 2)

        TB.reconcile_population!(mixed, [250.0])

        @test mixed ≈ reshape([50.0, 200.0], 1, 2)
    end

    @testset "Complete annual update" begin
        data = F.make_data()
        model = F.model()

        demo = TB.EquilibriumDemography(
            data, 2023; nages=3
        )

        u = F.state([
            100.0 0.0 4.0
            0.0 200.0 5.0
            0.0 300.0 6.0
        ])

        TB.apply_demography!(
            u, model, demo, 2023
        )

        U = reshape(u, 3, 3)

        @test vec(sum(@view(U[1:2, :]); dims=1)) ≈
            [100.0, 200.0, 300.0]

        @test collect(@view U[3, :]) ==
            [4.0, 5.0, 6.0]

        @test all(iszero, @view U[2, 1:2])
        @test U[1, 1] == 100.0
    end

    @testset "Projected population target" begin
        data = F.make_data()
        demo = TB.ProjectedDemography(data; nages=3)

        @test TB.target_population(demo, 2023) ≈
            [150.0, 250.0, 350.0]
    end

    @testset "Automatic differentiation" begin
        data = F.make_data()
        model = F.model()

        demo = TB.EquilibriumDemography(
            data, 2023; nages=3
        )

        function prediction(x)
            T = typeof(x)
            u = zeros(T, 9)

            # Age 0
            u[1] = T(100)

            # Age 1
            u[4] = x
            u[5] = T(100)

            # Age 2+
            u[7] = T(100)
            u[8] = T(200)

            TB.apply_demography!(
                u, model, demo, 2023
            )

            return u[7]
        end

        # After ageing:
        # age 2+ susceptible = x + 100
        # age 2+ infected = 300
        # target population = 300.
        #
        # prediction(x) = 300(x+100)/(x+400)

        @test prediction(100.0) ≈ 120.0

        derivative = ForwardDiff.derivative(
            prediction, 100.0
        )

        @test derivative ≈ 0.36 atol=1e-10
    end
end
