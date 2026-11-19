
using Test
using LinearAlgebra
using TBKiribatiJulia

const TB = TBKiribatiJulia

@testset "Shared model mixing" begin

    C = [
        2.0 1.0 0.0
        3.0 4.0 1.0
        0.0 2.0 5.0
    ]

    @testset "Contact normalisation" begin
        Ccopy = copy(C)
        N = TB.normalise_model_contacts(C)

        @test sum(N) / size(N, 1) ≈ 1.0
        @test N ≈ C ./ (sum(C) / 3)
        @test C == Ccopy

        # Normalisation is idempotent.
        @test TB.normalise_model_contacts(N) ≈ N

        @test_throws DimensionMismatch TB.normalise_model_contacts(ones(2, 3))

        @test_throws ArgumentError TB.normalise_model_contacts(zeros(3, 3))

        @test_throws ArgumentError TB.normalise_model_contacts(fill(-1.0, 3, 3))
    end

    @testset "Age-dependent infectiousness" begin
        q = TB.adult_infectiousness(96)

        @test length(q) == 96
        @test all(iszero, q[1:15])
        @test all(isone, q[16:96])

        @test all(iszero, TB.adult_infectiousness(10))

        @test TB.adult_infectiousness(
            5; minimum_age=2
        ) == [0.0, 0.0, 1.0, 1.0, 1.0]
    end

    @testset "Independent model construction" begin
        C20 = ones(20, 20)
        P = repeat(
            reshape([0.5, 0.3, 0.2], 1, 3),
            20, 1
        )

        dodd = TB.DoddTBModel(C20)
        extended = TB.DoddExtendedTBModel(C20, P)

        @test dodd.contact ≈ extended.contact
        @test dodd.infectiousness ==
            extended.infectiousness

        @test !hasproperty(extended, :reference)

        @test TB.state_layout(dodd).nepi == 6
        @test TB.state_layout(extended).nepi == 13
    end
end
