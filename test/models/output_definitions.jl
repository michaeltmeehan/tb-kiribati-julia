
using Test
using ForwardDiff
using TBKiribatiJulia

module OutputDefinitionsFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

"""
Construct synthetic saved states for two calendar years.

Counter rates are annual increments per age, scaled by:
- age index a
- scale

The second year has twice the first year's increment.
"""
function solution(
    model,
    counter_rates;
    stocks=Dict{Int,Float64}(),
    scale=1.0,
)
    layout = TB.state_layout(model)
    T = typeof(scale)

    function make_state(multiplier)
        u = zeros(T, TB.state_dimension(layout))

        for a in 1:layout.nages
            for (c, value) in stocks
                u[TB.state_index(layout, a, c)] =
                    scale * value * a
            end

            for (c, rate) in counter_rates
                u[TB.state_index(layout, a, c)] =
                    scale * multiplier * rate * a
            end
        end

        return u
    end

    u0 = make_state(0)
    u1 = make_state(1)
    u2 = copy(u1)
    u3 = make_state(3)

    return (
        t=[2023.0, 2024.0, 2024.0, 2025.0],
        u=[u0, u1, u2, u3],
    )
end

end


@testset "Epidemiological output definitions" begin
    TB = TBKiribatiJulia
    F = OutputDefinitionsFixture
    years = 2023:2024

    @testset "Dodd event definitions" begin
        model = TB.DoddTBModel(
            ones(2, 2);
            pulmonary_fraction=0.8,
        )

        rates = Dict(
            TB.CumFastProgression => 2.0,
            TB.CumSlowProgression => 3.0,
            TB.CumRelapse => 1.0,
            TB.CumTreatmentInitiation => 4.0,
            TB.CumUntreatedTBDeaths => 2.0,
            TB.CumTreatmentTBDeaths => 1.0,
        )

        sol = F.solution(
            model,
            rates;
            stocks=Dict(
                TB.E => 4.0,
                TB.L => 6.0,
                TB.I => 8.0,
                TB.T => 2.0,
            ),
        )

        out = TB.model_outputs(
            model, sol;
            variables=(
                :disease_onsets,
                :treatment_initiations,
                :tb_deaths,
                :treatment_deaths,
                :incidence,
                :notifications,
                :mortality,
                :untreated_active_prevalence,
                :latent_prevalence,
                :treatment_prevalence,
                :active_prevalence,
            ),
            years,
        )

        # Age 1: annual onset increments 6 and 12.
        # Age 2: annual onset increments 12 and 24.
        @test out.disease_onsets ≈ [
            6.0 12.0
            12.0 24.0
        ]

        @test out.treatment_initiations ≈ [
            4.0 8.0
            8.0 16.0
        ]

        @test out.tb_deaths ≈ [
            3.0 6.0
            6.0 12.0
        ]

        @test out.treatment_deaths ≈ [
            1.0 2.0
            2.0 4.0
        ]

        # Existing all-form extrapolations remain unchanged.
        @test out.incidence ≈
            out.disease_onsets ./ 0.8

        @test out.notifications ≈
            out.treatment_initiations ./ 0.8

        @test out.mortality ≈
            out.tb_deaths ./ 0.8

        # Stocks are evaluated at each year's start.
        @test out.untreated_active_prevalence ≈ [
            8.0 8.0
            16.0 16.0
        ]

        @test out.latent_prevalence ≈ [
            10.0 10.0
            20.0 20.0
        ]

        @test out.treatment_prevalence ≈ [
            2.0 2.0
            4.0 4.0
        ]

        @test out.active_prevalence ==
            out.untreated_active_prevalence
    end

    @testset "DoddExtended event identities" begin
        DE = TB.DoddExtendedStates

        P = repeat(
            reshape([0.5, 0.3, 0.2], 1, 3),
            2, 1,
        )

        model = TB.DoddExtendedTBModel(
            ones(2, 2), P
        )

        rates = Dict(
            DE.CumFastProgression => 3.0,
            DE.CumSlowProgression => 2.0,
            DE.CumRelapse => 1.0,
            DE.CumTreatmentInitiation => 10.0,
            DE.CumUntreatedTBDeaths => 2.0,
            DE.CumTreatmentTBDeaths => 1.0,
            DE.CumNewSP => 2.0,
            DE.CumNewSN => 1.0,
            DE.CumNewEP => 3.0,
            DE.CumRecurrentSP => 1.0,
            DE.CumRecurrentSN => 2.0,
            DE.CumRecurrentEP => 1.0,
        )

        stocks = Dict(
            DE.E0 => 1.0,
            DE.E1 => 2.0,
            DE.L0 => 3.0,
            DE.L1 => 4.0,
            DE.T => 5.0,
        )

        for c in DE.I0SP:DE.I1EP
            stocks[c] = 1.0
        end

        sol = F.solution(model, rates; stocks)

        out = TB.model_outputs(
            model, sol;
            variables=(
                :disease_onsets,
                :incidence,
                :treatment_initiations,
                :notifications,
                :new_notifications,
                :recurrent_notifications,
                :pulmonary_notifications,
                :extrapulmonary_notifications,
                :tb_deaths,
                :mortality,
                :untreated_mortality,
                :treatment_mortality,
                :treatment_deaths,
                :untreated_active_prevalence,
                :latent_prevalence,
                :treatment_prevalence,
                :active_prevalence,
            ),
            years,
        )

        @test out.disease_onsets == out.incidence
        @test out.treatment_initiations == out.notifications
        @test out.tb_deaths == out.mortality

        @test out.notifications ≈
            out.new_notifications .+
            out.recurrent_notifications

        @test out.notifications ≈
            out.pulmonary_notifications .+
            out.extrapulmonary_notifications

        @test out.tb_deaths ≈
            out.untreated_mortality .+
            out.treatment_mortality

        @test out.treatment_deaths ==
            out.treatment_mortality

        @test out.untreated_active_prevalence ≈ [
            6.0 6.0
            12.0 12.0
        ]

        @test out.latent_prevalence ≈ [
            10.0 10.0
            20.0 20.0
        ]

        @test out.treatment_prevalence ≈ [
            5.0 5.0
            10.0 10.0
        ]

        @test out.active_prevalence ==
            out.untreated_active_prevalence
    end

    @testset "ForwardDiff type preservation" begin
        model = TB.DoddTBModel(ones(2, 2))

        rates = Dict(
            TB.CumFastProgression => 2.0,
            TB.CumSlowProgression => 3.0,
            TB.CumRelapse => 1.0,
        )

        function prediction(x)
            sol = F.solution(
                model, rates; scale=x
            )

            out = TB.model_output(
                model,
                sol,
                :disease_onsets;
                years,
                age_breaks=[0, 2],
            )

            return sum(out)
        end

        # Counter rate 6, sum of age weights 3,
        # sum of yearly multipliers 3.
        # Therefore total = 54x.
        @test prediction(2.0) ≈ 108.0

        @test ForwardDiff.derivative(
            prediction, 2.0
        ) ≈ 54.0
    end
end
