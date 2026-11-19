
using Test
using TBKiribatiJulia

module EnsembleFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

struct TestModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

struct TestDemography <: TB.AbstractDemography
    annual_addition::Float64
end

TB.state_layout(m::TestModel) = m.layout

function TB.make_parameters(
    ::TestModel,
    draw::NamedTuple,
)
    draw.beta >= 0 ||
        throw(ArgumentError(
            "Transmission rate must be nonnegative"
        ))

    return draw
end

function TB.initial_state(
    ::TestModel,
    p,
    ::TestDemography;
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
    ::TestModel,
    d::TestDemography,
    year::Real;
    kwargs...,
)
    u[1] += d.annual_addition
    return nothing
end

TB.default_output_variables(::TestModel) =
    (:notifications, :disease_onsets)

function TB.model_output(
    ::TestModel,
    ex::TB.OutputExtraction,
    ::Val{:notifications},
)
    return TB.counter_output(ex, 3)
end

function TB.model_output(
    ::TestModel,
    ex::TB.OutputExtraction,
    ::Val{:disease_onsets},
)
    return TB.counter_output(ex, 3)
end

model() = TestModel(TB.StateLayout(2, 1, 1))
demography() = TestDemography(10.0)

function expected_events(beta)
    e = exp(-beta)

    first_year = 100 * (1 - e)
    second_year = (100 * e + 10) * (1 - e)

    return [first_year, second_year]
end

end # module


@testset "Prepared simulation ensembles" begin
    TB = TBKiribatiJulia
    F = EnsembleFixture

    model = F.model()
    demography = F.demography()

    setup = TB.prepare_simulation(
        model,
        demography;
        start_year=2023,
        end_year=2025,
        abstol=1e-11,
        reltol=1e-11,
    )

    draws = [
        (beta=0.1,),
        (beta=-1.0,),  # Deliberately invalid
        (beta=0.4,),
    ]

    @testset "Selective extraction and failures" begin
        result = TB.simulate_ensemble(
            setup,
            draws;
            output_years=2023:2024,
            variables=(:notifications,),
        )

        @test length(result.draws) == 3
        @test length(result.outputs) == 3
        @test result.draws == draws

        @test !isnothing(result.outputs[1])
        @test isnothing(result.outputs[2])
        @test !isnothing(result.outputs[3])

        @test length(result.failures) == 1
        @test only(result.failures).draw == 2
        @test occursin(
            "ArgumentError",
            only(result.failures).error_type,
        )

        for i in (1, 3)
            output = result.outputs[i]

            @test keys(output) ==
                (:years, :age_breaks, :notifications)

            @test output.years == [2023, 2024]
            @test size(output.notifications) == (1, 2)

            @test vec(output.notifications) ≈
                F.expected_events(draws[i].beta) atol=1e-7
        end

        @test !hasproperty(
            result.outputs[1], :disease_onsets
        )
    end

    @testset "Multiple requested variables" begin
        result = TB.simulate_ensemble(
            setup,
            [(beta=0.2,)];
            output_years=2023:2024,
            variables=(
                :notifications,
                :disease_onsets,
            ),
        )

        @test isempty(result.failures)

        output = only(result.outputs)

        @test output.notifications ==
            output.disease_onsets
    end

    @testset "Convenience interface" begin
        prepared = TB.simulate_ensemble(
            setup,
            [(beta=0.2,)];
            output_years=2023:2024,
            variables=(:notifications,),
        )

        direct = TB.simulate_ensemble(
            model,
            [(beta=0.2,)],
            demography;
            start_year=2023,
            end_year=2025,
            output_years=2023:2024,
            variables=(:notifications,),
            abstol=1e-11,
            reltol=1e-11,
        )

        @test only(prepared.outputs).notifications ≈
            only(direct.outputs).notifications
    end

    @testset "Ensemble summaries" begin
        result = TB.simulate_ensemble(
            setup,
            draws;
            output_years=2023:2024,
            variables=(:notifications,),
        )

        summary = TB.summarise_ensemble(
            result;
            variables=(:notifications,),
            probabilities=[0.5],
        )

        @test summary.n_draws == 3
        @test summary.n_success == 2
        @test summary.n_failed == 1
        @test summary.failure_rate ≈ 1/3

        @test size(summary.totals.notifications) == (2, 1)

        expected = (
            F.expected_events(0.1) .+
                F.expected_events(0.4)
        ) ./ 2

        @test vec(summary.totals.notifications) ≈
            expected atol=1e-7
    end

    @testset "Invalid requests" begin
        @test_throws ArgumentError TB.simulate_ensemble(
            setup,
            draws;
            output_years=2025:2025,
        )

        @test_throws ArgumentError TB.simulate_ensemble(
            setup,
            draws;
            output_years=2023:2024,
            variables=(),
        )
    end
end
