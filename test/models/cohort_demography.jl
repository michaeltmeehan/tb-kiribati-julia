
using Test
using ForwardDiff
using TBKiribatiJulia

module CohortFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

struct TestModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

TB.state_layout(model::TestModel) = model.layout

function data(; national_migration=0.0)
    return TB.DemographicData(
        # Population stored in thousands.
        [0.1 0.1;
            0.2 0.2;
            0.3 0.3],
        # Fertility applies to age 1.
        reshape([0.2, 0.2], 1, 2),
        zeros(3, 2),
        [0.0, national_migration / 1000],
        2023,
        2024,
        1,
        0.485,
    )
end

end


@testset "Independent cohort demography" begin
    TB = TBKiribatiJulia
    F = CohortFixture

    @testset "Births and ageing" begin
        model = F.TestModel(
            TB.StateLayout(1, 0, 3)
        )

        demo = TB.CohortDemography(
            F.data();
            nages=3,
            female_fraction=0.5,
        )

        u = [100.0, 200.0, 300.0]

        TB.apply_demography!(
            u, model, demo, 2023
        )

        # Before births:
        # age 0 = 0
        # age 1 = 100
        # age 2+ = 500
        #
        # Fertility exposure at age 1:
        # (200 + 100) / 2 = 150
        #
        # Births = 0.5 * 0.2 * 150 = 15

        @test u ≈ [15.0, 100.0, 500.0]
        @test sum(u) ≈ 615.0
    end

    @testset "Age-specific migration" begin
        model = F.TestModel(
            TB.StateLayout(1, 0, 3)
        )

        # Net migration sums to zero.
        M = reshape([-5.0, 10.0, -5.0], 3, 1)

        demo = TB.CohortDemography(
            F.data();
            nages=3,
            migration_scheme=:age,
            age_migration=M,
            migration_first_year=2024,
        )

        u = [100.0, 200.0, 300.0]

        TB.apply_demography!(
            u, model, demo, 2023
        )

        # Fertility exposure at age 1:
        # (200 + 110) / 2 = 155
        #
        # Births = 15.5
        #
        # Migration: [-5, +10, -5]

        @test u ≈ [10.5, 110.0, 495.0]
        @test sum(u) ≈ 615.5
    end

    @testset "Disease states and counters" begin
        model = F.TestModel(
            TB.StateLayout(2, 1, 3)
        )

        demo = TB.CohortDemography(
            F.data();
            nages=3,
        )

        # Age-major: susceptible, infected, counter.
        u = [
            100.0, 5.0, 9.0,
            200.0, 10.0, 8.0,
            300.0, 20.0, 7.0,
        ]

        original_counters = u[[3, 6, 9]]

        TB.apply_demography!(
            u, model, demo, 2023
        )

        U = reshape(u, 3, 3)

        # All newborns enter susceptible.
        @test U[2, 1] == 0.0

        # Existing infected individuals age normally.
        @test U[2, 2] ≈ 5.0
        @test U[2, 3] ≈ 30.0

        # Cumulative counters are unaffected.
        @test collect(U[3, :]) ==
            original_counters

        @test all(u .>= 0)
    end

    @testset "Proportional emigration" begin
        model = F.TestModel(
            TB.StateLayout(1, 0, 3)
        )

        demo = TB.CohortDemography(
            F.data(; national_migration=-60.0);
            nages=3,
        )

        u = [100.0, 200.0, 300.0]

        TB.apply_demography!(
            u, model, demo, 2023
        )

        # Migration is proportional to the post-ageing
        # population and sums to -60.
        @test sum(u) < 615.0
        @test all(u .>= 0)
    end

    @testset "Differentiation" begin
        model = F.TestModel(
            TB.StateLayout(1, 0, 3)
        )

        demo = TB.CohortDemography(
            F.data(); nages=3
        )

        function projected_total(x)
            u = [one(x)*100, x, one(x)*300]

            TB.apply_demography!(
                u, model, demo, 2023
            )

            return sum(u)
        end

        # Initial x contributes directly to the
        # surviving population and to midyear births.
        #
        # d(total)/dx = 1 + (0.5 * 0.2 / 2)
        #             = 1.05

        @test ForwardDiff.derivative(
            projected_total, 200.0
        ) ≈ 1.05 atol=1e-10
    end
end
