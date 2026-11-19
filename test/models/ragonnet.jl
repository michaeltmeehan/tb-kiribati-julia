
using Test
using ForwardDiff
using SciMLBase: successful_retcode
using TBKiribatiJulia

module RagonnetFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

const IDX = TB.RagonnetIndices

function demography()
    data = TB.DemographicData(
        fill(1.0, 101, 3),
        zeros(45, 3),
        fill(0.01, 101, 3),
        zeros(3),
        2023,
        2025,
        10,
        0.485,
    )

    return TB.ProjectedDemography(data; nages=96)
end

model() = TB.RagonnetTBModel(ones(96, 96))

function seeded_state(model, demography)
    params = TB.make_parameters(model; beta=0.0)

    u = TB.initial_state(
        model, params, demography;
        year=2023,
    )

    layout = TB.state_layout(model)

    # Seed untreated infectious disease without changing
    # the living population.
    for age in 16:96
        s = TB.state_index(layout, age, IDX.MtbNaive)
        i = TB.state_index(layout, age, IDX.ClinInf)

        u[s] -= 1.0
        u[i] += 1.0
    end

    return u
end

function living(u, layout)
    return [
        sum(
            u[TB.state_index(layout, a, c)]
            for c in 1:layout.nepi
        )
        for a in 1:layout.nages
    ]
end

end


@testset "Independent Ragonnet model" begin
    TB = TBKiribatiJulia
    F = RagonnetFixture
    IDX = F.IDX

    model = F.model()
    demography = F.demography()
    layout = TB.state_layout(model)

    @testset "Model structure" begin
        @test layout.nepi == 10
        @test layout.ncum == 7
        @test layout.nages == 96
        @test TB.state_dimension(layout) == 1632

        # Ragonnet contacts are deliberately unnormalised.
        @test model.contact == ones(96, 96)
        @test !hasproperty(model, :reference)
    end

    @testset "Initialisation" begin
        params = TB.make_parameters(model; beta=0.0)

        u0 = TB.initial_state(
            model, params, demography;
            year=2023,
        )

        @test length(u0) == 1632
        @test all(isfinite, u0)
        @test all(x -> x >= 0, u0)

        @test F.living(u0, layout) ≈
            TB.get_population(demography, 2023)

        for a in 1:96
            for c in 11:17
                @test iszero(
                    u0[TB.state_index(layout, a, c)]
                )
            end
        end
    end

    @testset "RHS conservation" begin
        params = TB.make_parameters(
            model;
            beta=0.002,
            disease_mortality_clin_inf=0.1,
        )

        u = F.seeded_state(model, demography)
        du = similar(u)

        ctx = TB.SimulationContext(
            model, params, demography,
            TB.TBWorkspace(96),
        )

        TB.tb_rhs!(du, u, ctx, 2023.0)

        @test all(isfinite, du)

        D = reshape(du, 17, 96)

        for a in 1:96
            @test sum(@view D[1:10, a]) +
                D[IDX.CumDeathsTB, a] ≈
                0.0 atol=1e-9
        end

        # Ages under 15 do not contribute infectiousness.
        @test all(iszero,
            ctx.workspace.infectious_prevalence[1:15]
        )
    end

    @testset "Simulation and outputs" begin
        params = TB.make_parameters(
            model;
            beta=0.002,
            disease_mortality_clin_inf=0.1,
        )

        setup = TB.prepare_simulation(
            model, demography;
            start_year=2023,
            end_year=2025,
            abstol=1e-9,
            reltol=1e-8,
        )

        sol = TB.simulate(
            setup, params;
            u0=F.seeded_state(model, demography),
        )

        @test successful_retcode(sol)

        out = TB.model_outputs(
            model, sol;
            variables=(
                :infection_events,
                :disease_onsets,
                :progression_to_active_tb,
                :treatment_relapses,
                :treatment_initiations,
                :notifications,
                :tb_deaths,
                :untreated_active_prevalence,
                :treatment_prevalence,
            ),
            years=2023:2024,
            age_breaks=[0, 5, 15, 25, 35,
                45, 55, 65, 96],
        )

        @test size(out.disease_onsets) == (8, 2)

        @test out.disease_onsets ≈
            out.progression_to_active_tb .+
            out.treatment_relapses

        @test out.notifications ==
            out.treatment_initiations

        @test all(isfinite, out.tb_deaths)

        # The end-of-year demographic updates restore
        # the population targets while preserving counters.
        for year in (2024, 2025)
            ii = findall(t -> isapprox(t, year), sol.t)

            @test length(ii) >= 2

            before = sol.u[first(ii)]
            after = sol.u[last(ii)]

            @test F.living(after, layout) ≈
                TB.get_population(demography, year)

            for a in 1:96, c in 11:17
                k = TB.state_index(layout, a, c)
                @test before[k] ≈ after[k] atol=1e-8
            end
        end
    end

    @testset "ForwardDiff through full simulation" begin
        setup = TB.prepare_simulation(
            model, demography;
            start_year=2023,
            end_year=2024,
            abstol=1e-9,
            reltol=1e-8,
        )

        u0 = F.seeded_state(model, demography)

        function prediction(beta)
            params = TB.make_parameters(model; beta)

            sol = TB.simulate(setup, params; u0)

            @assert successful_retcode(sol)

            return sum(TB.model_output(
                model, sol, :disease_onsets;
                years=2023:2023,
            ))
        end

        beta0 = 0.002
        h = 1e-5

        gradient_ad = ForwardDiff.derivative(
            prediction, beta0
        )

        gradient_fd = (
            prediction(beta0 + h) -
                prediction(beta0 - h)
        ) / (2h)

        @test isfinite(gradient_ad)
        @test gradient_ad ≈ gradient_fd rtol=5e-3 atol=1e-3
    end

    @testset "Equilibrium interface" begin
        params = TB.make_parameters(model)

        @test isnothing(
            TB.equilibrium_parameter_check(model, params)
        )
    end
end
