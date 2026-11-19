
using Test
using ForwardDiff
using SciMLBase: successful_retcode
using TBKiribatiJulia

module PreparedSimulationFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

struct TestModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

struct TestDemography <: TB.AbstractDemography
    annual_births::Float64
end

TB.state_layout(m::TestModel) = m.layout

TB.make_parameters(
    m::TestModel,
    draw::NamedTuple,
) = draw

function TB.initial_state(
    m::TestModel,
    p,
    d::TestDemography;
    year::Real,
)
    T = typeof(p.beta)
    return T[100, 0, 0]
end

function TB.tb_rhs!(
    du,
    u,
    ctx::TB.SimulationContext{<:TestModel},
    t,
)
    flow = ctx.params.beta * u[1]

    du[1] = -flow
    du[2] = flow
    du[3] = flow

    return nothing
end

function TB.apply_demography!(
    u::AbstractVector,
    m::TestModel,
    d::TestDemography,
    year::Real;
    kwargs...,
)
    u[1] += d.annual_births
    return nothing
end

model() = TestModel(TB.StateLayout(2, 1, 1))
demography() = TestDemography(10.0)

# Analytical cumulative infections after one year,
# an annual addition of 10 susceptibles, and another
# half-year of transmission.
expected_counter(beta) =
    100 * (1 - exp(-beta)) +
        (100 * exp(-beta) + 10) *
        (1 - exp(-beta / 2))

expected_gradient(beta) =
    150 * exp(-1.5 * beta) +
        5 * exp(-0.5 * beta)

end


@testset "Prepared simulation" begin
    TB = TBKiribatiJulia
    F = PreparedSimulationFixture

    model = F.model()
    demo = F.demography()

    setup = TB.prepare_simulation(
        model,
        demo;
        start_year=2023,
        end_year=2024.5,
        abstol=1e-11,
        reltol=1e-11,
        saveat=[2023.5, 2024.5],
    )

    @testset "Repeated parameter proposals" begin
        @test setup.model === model
        @test setup.demography === demo

        for beta in (0.05, 0.2, 0.8)
            sol = TB.simulate(
                setup, (beta=beta,)
            )

            @test successful_retcode(sol)
            @test sol.t[end] ≈ 2024.5

            @test sol.u[end][3] ≈
                F.expected_counter(beta) atol=1e-7
        end
    end

    @testset "Independent solutions" begin
        sol1 = TB.simulate(
            setup, (beta=0.1,)
        )

        sol2 = TB.simulate(
            setup, (beta=0.4,)
        )

        @test sol1.u[end][3] != sol2.u[end][3]

        # A repeated proposal must give the same result.
        sol3 = TB.simulate(
            setup, (beta=0.1,)
        )

        @test sol3.u[end] ≈ sol1.u[end]
    end

    @testset "Annual callback" begin
        sol = TB.simulate(
            setup, (beta=0.2,)
        )

        indices = findall(
            t -> isapprox(t, 2024.0; atol=1e-8),
            sol.t,
        )

        @test length(indices) >= 2

        before = sol.u[first(indices)]
        after = sol.u[last(indices)]

        @test after[1] - before[1] ≈ 10.0 atol=1e-7
        @test after[2] ≈ before[2] atol=1e-7
        @test after[3] ≈ before[3] atol=1e-7
    end

    @testset "Convenience interface" begin
        direct = TB.simulate(
            model,
            (beta=0.2,),
            demo;
            u0=[100.0, 0.0, 0.0],
            tspan=(2023.0, 2024.5),
            abstol=1e-11,
            reltol=1e-11,
            saveat=[2023.5, 2024.5],
        )

        prepared = TB.simulate(
            setup, (beta=0.2,)
        )

        @test direct.u[end] ≈
            prepared.u[end] atol=1e-9

        drawn = TB.simulate_from_draw(
            setup, (beta=0.2,)
        )

        @test drawn.u[end] ≈
            prepared.u[end] atol=1e-9
    end

    @testset "Automatic differentiation" begin
        prediction(beta) = begin
            sol = TB.simulate(
                setup, (beta=beta,)
            )
            @assert successful_retcode(sol)
            return sol.u[end][3]
        end

        for beta in (0.1, 0.2, 0.5)
            gradient = ForwardDiff.derivative(
                prediction, beta
            )

            @test gradient ≈
                F.expected_gradient(beta) rtol=1e-5
        end
    end

    @testset "Invalid state dimensions" begin
        @test_throws DimensionMismatch TB.simulate(
            setup,
            (beta=0.2,);
            u0=zeros(2),
        )
    end
end
