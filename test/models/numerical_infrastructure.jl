
using Test
using TBKiribatiJulia
using LinearAlgebra

const TB = TBKiribatiJulia

@testset "Numerical infrastructure" begin

    @testset "StateLayout" begin
        layout = TB.StateLayout(6, 10, 96)

        @test layout.nepi == 6
        @test layout.ncum == 10
        @test layout.nages == 96

        @test TB.nstates(layout) == 16

        @test TB.state_index(layout, 1, 1) == 1
        @test TB.state_index(layout, 2, 1) == 17
        @test TB.state_index(layout, 96, 16) == 1536

        @test_throws ArgumentError TB.StateLayout(0, 10, 96)
        @test_throws ArgumentError TB.StateLayout(6, -1, 96)
        @test_throws ArgumentError TB.StateLayout(6, 10, 0)
    end

    @testset "Typed workspace" begin
        for T in (Float32, Float64, BigFloat)
            workspace = TB.TBWorkspace(T, 96)

            @test eltype(workspace.foi) == T
            @test eltype(workspace.infectious_prevalence) == T

            @test length(workspace.foi) == 96
            @test length(workspace.infectious_prevalence) == 96

            @test all(iszero, workspace.foi)
            @test all(iszero, workspace.infectious_prevalence)

            workspace.foi[1] = T(1.25)
            workspace.infectious_prevalence[1] = T(0.4)

            @test workspace.foi[1] == T(1.25)
            @test workspace.infectious_prevalence[1] == T(0.4)

            # Independent mutable arrays.
            @test workspace.foi !==
                workspace.infectious_prevalence
        end

        @test TB.TBWorkspace(96) isa TB.TBWorkspace{Float64}

        @test_throws ArgumentError TB.TBWorkspace(Float64, 0)
        @test_throws ArgumentError TB.TBWorkspace(BigFloat, -1)
    end

    @testset "SimulationContext" begin
        contact = Matrix{Float64}(I, 4, 4)
        model = TB.DoddTBModel(contact)

        for T in (Float32, Float64, BigFloat)
            workspace = TB.TBWorkspace(T, 4)

            params = (
                beta=T(0.5),
                progression=T(0.1),
            )

            ctx = TB.SimulationContext(
                model,
                params,
                nothing,
                workspace,
            )

            @test ctx.model === model
            @test ctx.params === params
            @test ctx.demography === nothing
            @test ctx.workspace === workspace

            @test eltype(ctx.workspace.foi) == T
            @test typeof(ctx.params.beta) == T
        end
    end
end
