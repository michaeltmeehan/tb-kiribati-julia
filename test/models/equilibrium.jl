
using Test
using TBKiribatiJulia

module EquilibriumFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

struct TestModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

TB.state_layout(m::TestModel) = m.layout

TB.equilibrium_parameter_check(
    ::TestModel,
    p::NamedTuple,
) = begin
    iszero(p.trend) ||
        throw(ArgumentError("Nonstationary parameter"))
    nothing
end

function TB.tb_rhs!(
    du,
    u,
    ctx::TB.SimulationContext{<:TestModel},
    t,
)
    beta = ctx.params.beta
    gamma = ctx.params.gamma

    fill!(du, zero(eltype(du)))

    for a in 1:2
        b = (a-1)*3

        infection = beta*u[b+1]
        recovery = gamma*u[b+2]

        du[b+1] = -infection + recovery
        du[b+2] = infection - recovery
        du[b+3] = infection
    end

    return nothing
end

# Isolate numerical equilibration from demographic changes.
function TB.apply_demography!(
    u::AbstractVector,
    ::TestModel,
    ::TB.EquilibriumDemography,
    year::Real;
    kwargs...,
)
    return nothing
end

function TB.model_output(
    ::TestModel,
    ex::TB.OutputExtraction,
    ::Val{:disease_onsets},
)
    return TB.counter_output(ex, 3)
end

function fixture()
    data = TB.DemographicData(
        [
            0.1 0.1
            0.2 0.2
        ],
        zeros(1, 2),
        zeros(2, 2),
        zeros(2),
        2023,
        2024,
        10,
        0.485,
    )

    demography = TB.EquilibriumDemography(
        data, 2023; nages=2
    )

    model = TestModel(TB.StateLayout(2, 1, 2))

    params = (
        beta=0.2,
        gamma=0.3,
        trend=0.0,
    )

    u0 = [
        100.0, 0.0, 0.0,
        200.0, 0.0, 0.0,
    ]

    return (; model, params, demography, u0)
end

end


@testset "Equilibrium simulation" begin
    TB = TBKiribatiJulia
    F = EquilibriumFixture

    (; model, params, demography, u0) = F.fixture()

    @testset "Known analytical equilibrium" begin
        result = TB.equilibrate(
            model,
            params,
            demography;
            u0,
            start_year=2023,
            block_years=7,
            max_years=100,
            state_tol=1e-7,
            incidence_tol=1e-7,
            abstol=1e-10,
            reltol=1e-10,
        )

        @test result.converged
        @test result.years <= 100
        @test result.state_change < 1e-7
        @test result.incidence_change < 1e-7

        # Equilibrium S = 0.6N, I = 0.4N.
        @test result.u[1] ≈ 60.0 atol=1e-4
        @test result.u[2] ≈ 40.0 atol=1e-4
        @test result.u[4] ≈ 120.0 atol=1e-4
        @test result.u[5] ≈ 80.0 atol=1e-4

        # Annual incidence at equilibrium:
        # beta * (60 + 120) = 36.
        @test result.incidence ≈ 36.0 atol=1e-3

        # All cumulative counters are reset.
        @test result.u[3] == 0
        @test result.u[6] == 0
    end

    @testset "Initial-condition independence" begin
        result = TB.equilibrate(
            model,
            params,
            demography;
            u0=[
                0.0, 100.0, 0.0,
                0.0, 200.0, 0.0,
            ],
            max_years=100,
            state_tol=1e-7,
            incidence_tol=1e-7,
        )

        @test result.converged
        @test result.u[1] ≈ 60.0 atol=1e-3
        @test result.u[4] ≈ 120.0 atol=1e-3
    end

    @testset "Nonconvergence" begin
        result = TB.equilibrate(
            model,
            params,
            demography;
            u0,
            max_years=1,
            block_years=1,
            state_tol=1e-12,
            incidence_tol=1e-12,
        )

        @test !result.converged
        @test result.years == 1
    end

    @testset "Stationarity validation" begin
        trending = merge(
            params, (trend=0.01,)
        )

        @test_throws ArgumentError TB.equilibrate(
            model,
            trending,
            demography;
            u0,
        )

        @test_throws ArgumentError TB.equilibrate(
            model,
            params,
            demography;
            u0,
            block_years=0,
        )

        @test_throws DimensionMismatch TB.equilibrate(
            model,
            params,
            demography;
            u0=zeros(4),
        )
    end

    @testset "Frozen demographic mortality" begin
        @test TB.demographic_mortality_year(
            demography, 2050
        ) == 2023
    end
end
