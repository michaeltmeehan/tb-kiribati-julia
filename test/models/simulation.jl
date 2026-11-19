
using Test
using ForwardDiff
using SciMLBase: successful_retcode
using TBKiribatiJulia

module SimulationFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

struct TestModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

struct TestDemography <: TB.AbstractDemography
    annual_addition::Float64
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
    infection = ctx.params.beta * u[1]

    du[1] = -infection
    du[2] = infection
    du[3] = infection

    return nothing
end

function TB.apply_demography!(
    u::AbstractVector,
    m::TestModel,
    d::TestDemography,
    year::Real;
    kwargs...,
)
    u[1] += d.annual_addition
    return nothing
end

model() = TestModel(TB.StateLayout(2, 1, 1))
demography() = TestDemography(10.0)

end


@testset "Shared simulation engine" begin
    TB = TBKiribatiJulia
    F = SimulationFixture

    model = F.model()
    demo = F.demography()

    beta = 0.2
    S0 = 100.0

    @testset "Continuous dynamics" begin
        # No annual demographic update occurs.
        sol = TB.simulate(
            model,
            (beta=beta,),
            demo;
            u0=[S0, 0.0, 0.0],
            tspan=(2023.0, 2023.5),
            saveat=[2023.25, 2023.5],
            abstol=1e-11,
            reltol=1e-11,
        )

        @test successful_retcode(sol)

        expected_S = S0 * exp(-0.5 * beta)
        expected_C = S0 - expected_S

        @test sol.u[end][1] ≈ expected_S atol=1e-7
        @test sol.u[end][2] ≈ expected_C atol=1e-7
        @test sol.u[end][3] ≈ expected_C atol=1e-7

        @test any(t -> isapprox(t, 2023.25), sol.t)
        @test any(t -> isapprox(t, 2023.5), sol.t)
    end

    @testset "Annual demographic callbacks" begin
        sol = TB.simulate(
            model,
            (beta=beta,),
            demo;
            u0=[S0, 0.0, 0.0],
            tspan=(2023.0, 2025.2),
            saveat=[2023.5, 2024.5, 2025.2],
            abstol=1e-11,
            reltol=1e-11,
        )

        @test successful_retcode(sol)

        for year in (2024.0, 2025.0)
            indices = findall(
                t -> isapprox(t, year; atol=1e-8),
                sol.t,
            )

            @test length(indices) >= 2

            before = sol.u[first(indices)]
            after = sol.u[last(indices)]

            # Only susceptible population changes.
            @test after[1] - before[1] ≈ 10.0 atol=1e-7
            @test after[2] ≈ before[2] atol=1e-7
            @test after[3] ≈ before[3] atol=1e-7
        end

        @test sol.t[end] ≈ 2025.2

        # Living population changes only through
        # the annual population additions.
        @test sum(sol.u[end][1:2]) ≈ 120.0 atol=1e-7
    end

    @testset "Annual event extraction" begin
        sol = TB.simulate(
            model,
            (beta=beta,),
            demo;
            u0=[S0, 0.0, 0.0],
            tspan=(2023.0, 2025.0),
            abstol=1e-11,
            reltol=1e-11,
        )

        events = TB.annual_events(
            sol,
            model,
            3;
            years=2023:2024,
        )

        first_year = S0 * (1 - exp(-beta))

        second_year = (
            S0 * exp(-beta) + 10
        ) * (1 - exp(-beta))

        @test size(events) == (1, 2)
        @test events[1, 1] ≈ first_year atol=1e-7
        @test events[1, 2] ≈ second_year atol=1e-7
    end

    @testset "Automatic differentiation" begin
        function prediction(b)
            sol = TB.simulate(
                model,
                (beta=b,),
                demo;
                u0=[100.0, 0.0, 0.0],
                tspan=(2023.0, 2024.5),
                abstol=1e-11,
                reltol=1e-11,
            )

            @assert successful_retcode(sol)
            return sol.u[end][3]
        end

        # First year: 100(1-exp(-beta))
        # Following half-year:
        # (100exp(-beta)+10)(1-exp(-beta/2))

        expected(b) =
            100 * (1 - exp(-b)) +
                (100 * exp(-b) + 10) *
                (1 - exp(-b / 2))

        @test prediction(beta) ≈ expected(beta) atol=1e-7

        gradient = ForwardDiff.derivative(
            prediction, beta
        )

        # Analytical derivative.
        expected_gradient =
            150 * exp(-1.5 * beta) +
                5 * exp(-0.5 * beta)

        @test gradient ≈
            expected_gradient rtol=1e-5
    end
end
