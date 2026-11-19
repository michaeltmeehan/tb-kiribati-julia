
using Test
using ForwardDiff
using TBKiribatiJulia

const TB = TBKiribatiJulia

@testset "Selective epidemiological outputs" begin

    # Two age groups, two years, with a demographic
    # discontinuity at 2024.
    #
    # State vectors are constructed directly to test
    # extraction independently of the ODE solver.

    function synthetic_solution(layout; counter, increment)
        stride = TB.nstates(layout)
        nages = layout.nages

        function state(cumulative)
            u = zeros(typeof(cumulative), stride * nages)

            for a in 1:nages
                b = (a-1)*stride
                u[b+1] = 100 * a
                u[b+counter] = cumulative * a
            end

            return u
        end

        return (
            t=[2023.0, 2024.0, 2024.0, 2025.0],
            u=[
                state(0 * increment),
                state(increment),
                state(increment),
                state(3 * increment),
            ],
        )
    end

    @testset "Annual counter extraction" begin
        layout = TB.StateLayout(2, 1, 2)

        sol = synthetic_solution(
            layout;
            counter=3,
            increment=5.0,
        )

        windows = TB.annual_windows(
            sol; years=2023:2024
        )

        events = TB.annual_events(
            sol, layout, 3, windows
        )

        @test events ≈ [
            5.0 10.0
            10.0 20.0
        ]

        @test size(events) == (2, 2)
        @test eltype(events) == Float64
    end

    @testset "Age aggregation" begin
        values = [
            1.0 2.0
            3.0 4.0
            5.0 6.0
            7.0 8.0
        ]

        aggregated = TB.aggregate_age_groups(
            values, [0, 2, 4]
        )

        @test aggregated == [
            4.0 6.0
            12.0 14.0
        ]

        @test_throws ArgumentError TB.aggregate_age_groups(
            values, [0, 3, 2, 4]
        )
    end

    @testset "Dodd selective extraction" begin
        model = TB.DoddTBModel(
            ones(2, 2);
            pulmonary_fraction=0.8,
        )

        layout = TB.state_layout(model)

        sol = synthetic_solution(
            layout;
            counter=TB.CumTreatmentInitiation,
            increment=8.0,
        )

        result = TB.model_outputs(
            model,
            sol;
            variables=(:notifications,),
            years=2023:2024,
            age_breaks=[0, 2],
        )

        @test keys(result) ==
            (:years, :age_breaks, :notifications)

        @test result.notifications ≈
            reshape([30.0, 60.0], 1, 2)

        @test result.age_breaks == [0, 2]
    end

    @testset "DoddExtended selective extraction" begin
        P = repeat(
            reshape([0.5, 0.3, 0.2], 1, 3),
            2, 1
        )

        model = TB.DoddExtendedTBModel(
            ones(2, 2), P
        )

        layout = TB.state_layout(model)

        sol = synthetic_solution(
            layout;
            counter=TB.DoddExtendedStates.CumNewSP,
            increment=5.0,
        )

        result = TB.model_outputs(
            model,
            sol;
            variables=(
                :new_smear_positive,
                :new_notifications,
            ),
            years=2023:2024,
        )

        @test result.new_smear_positive ≈ [
            5.0 10.0
            10.0 20.0
        ]

        @test result.new_notifications ==
            result.new_smear_positive

        @test :mortality ∉ keys(result)

        @test_throws ArgumentError TB.model_output(
            model, sol, :unknown_output;
            years=2023:2024,
        )
    end

    @testset "ForwardDiff through output extraction" begin
        layout = TB.StateLayout(2, 1, 2)

        function prediction(x)
            sol = synthetic_solution(
                layout;
                counter=3,
                increment=x,
            )

            events = TB.annual_events(
                sol, layout, 3;
                years=2023:2024,
            )

            grouped = TB.aggregate_age_groups(
                events, [0, 2]
            )

            return sum(grouped)
        end

        # Age weights 1+2, event multipliers 1+2.
        # Total = 9x.
        @test prediction(2.0) ≈ 18.0

        @test ForwardDiff.derivative(
            prediction, 2.0
        ) ≈ 9.0
    end
end
