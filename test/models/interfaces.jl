
using Test
using TBKiribatiJulia

module ModelContractFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

struct TestModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

struct TestDemography <: TB.AbstractDemography
    population::Vector{Float64}
end

TB.state_layout(model::TestModel) = model.layout

TB.get_population(
    d::TestDemography,
    year::Real,
) = copy(d.population)

TB.make_parameters(
    model::TestModel,
    draw::NamedTuple,
) = draw

function TB.initial_state(
    model::TestModel,
    params,
    demography::TestDemography;
    year::Real,
)
    layout = TB.state_layout(model)

    u0 = zeros(
        typeof(params.beta),
        TB.state_dimension(layout),
    )

    for a in 1:layout.nages
        b = (a - 1) * TB.nstates(layout)

        N = demography.population[a]

        u0[b+1] = N
        u0[b+2] = zero(N)
        u0[b+3] = zero(N)
        u0[b+4] = zero(N)
    end

    return u0
end

function TB.tb_rhs!(
    du,
    u,
    ctx::TB.SimulationContext{<:TestModel},
    t,
)
    layout = TB.state_layout(ctx.model)
    beta = ctx.params.beta
    gamma = ctx.params.gamma

    fill!(du, zero(eltype(du)))

    for a in 1:layout.nages
        b = (a - 1) * TB.nstates(layout)

        S = u[b+1]
        I = u[b+2]

        infection = beta * S
        recovery = gamma * I

        du[b+1] = -infection
        du[b+2] = infection - recovery
        du[b+3] = recovery
        du[b+4] = infection
    end

    return nothing
end

end # module


@testset "Shared model interface" begin

    TB = TBKiribatiJulia
    Fixture = ModelContractFixture

    @testset "State layout" begin
        layout = TB.StateLayout(3, 1, 4)

        @test TB.nstates(layout) == 4
        @test TB.state_dimension(layout) == 16

        @test TB.state_index(layout, 1, 1) == 1
        @test TB.state_index(layout, 2, 1) == 5
        @test TB.state_index(layout, 4, 4) == 16

        @test_throws ArgumentError TB.StateLayout(0, 1, 4)
        @test_throws ArgumentError TB.StateLayout(3, -1, 4)
        @test_throws ArgumentError TB.StateLayout(3, 1, 0)
    end

    @testset "Synthetic model contract" begin
        model = Fixture.TestModel(
            TB.StateLayout(3, 1, 4)
        )

        demography = Fixture.TestDemography(
            [100.0, 200.0, 300.0, 400.0]
        )

        draw = (beta=0.02, gamma=0.5)

        params = TB.make_parameters(model, draw)

        @test params == draw
        @test TB.state_layout(model).nages == 4

        @test TB.validate_model_interface(
            model, params, demography;
            year=2023,
        )

        u0 = TB.initial_state(
            model, params, demography;
            year=2023,
        )

        @test length(u0) == 16
        @test sum(u0) == 1000.0

        ctx = TB.SimulationContext(
            model,
            params,
            demography,
            TB.TBWorkspace(4),
        )

        du = zeros(length(u0))

        TB.tb_rhs!(du, u0, ctx, 2023.0)

        D = reshape(du, 4, 4)

        # Exact expected initial infection flows.
        @test D[1, :] ≈
            [-2.0, -4.0, -6.0, -8.0]

        @test D[2, :] ≈
            [2.0, 4.0, 6.0, 8.0]

        @test all(iszero, D[3, :])

        @test D[4, :] ≈
            [2.0, 4.0, 6.0, 8.0]

        # Conservation of living population.
        @test all(iszero,
            vec(sum(@view(D[1:3, :]); dims=1))
        )

        # Counters are excluded from living population.
        @test sum(@view D[1:3, :]) == 0.0
        @test sum(@view D[4, :]) == 20.0
    end

    @testset "Numerical type preservation" begin
        model = Fixture.TestModel(
            TB.StateLayout(3, 1, 4)
        )

        demography = Fixture.TestDemography(
            [100.0, 200.0, 300.0, 400.0]
        )

        params = (
            beta=big"0.02",
            gamma=big"0.5",
        )

        u0 = TB.initial_state(
            model, params, demography;
            year=2023,
        )

        @test eltype(u0) == BigFloat

        @test TB.validate_model_interface(
            model, params, demography;
            year=2023,
        )
    end
end
