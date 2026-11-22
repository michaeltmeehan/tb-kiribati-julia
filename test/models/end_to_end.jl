
using Test
using ForwardDiff
using LinearAlgebra
using SciMLBase: successful_retcode
using TBKiribatiJulia

module EndToEndFixture

import ..TBKiribatiJulia
const TB = TBKiribatiJulia

const NAGES = 96
const FIRST_YEAR = 2023
const LAST_YEAR = 2026

function demographic_data()
    # 101 single-year ages, 2023–2026.
    # Population units in DemographicData are thousands.
    base = [
        0.9 + 0.005 * a for a in 0:100
    ]

    population = hcat(
        (base .* scale for scale in (
            1.00, 1.01, 1.02, 1.03
        ))...
    )

    mortality = fill(0.008, 101, 4)
    fertility = zeros(45, 4)
    migration = zeros(4)

    return TB.DemographicData(
        population,
        fertility,
        mortality,
        migration,
        FIRST_YEAR,
        LAST_YEAR,
        10,
        0.485,
    )
end

function demographic_scenario()
    return TB.ProjectedDemography(
        demographic_data();
        nages=NAGES,
    )
end

function contact_matrix()
    # Symmetric, nonzero contacts between every age.
    # The model constructor performs normalisation.
    C = Matrix{Float64}(undef, NAGES, NAGES)

    for a in 1:NAGES, b in 1:NAGES
        C[a, b] = 1 / (1 + abs(a - b) / 10)
    end

    return C
end

function presentation_matrix()
    return repeat(
        reshape([0.5, 0.3, 0.2], 1, 3),
        NAGES, 1,
    )
end

dodd_model() = TB.DoddTBModel(contact_matrix())

extended_model() = TB.DoddExtendedTBModel(
    contact_matrix(),
    presentation_matrix(),
)

function draw(beta; extended=false)
    base = (
        Lambda0=0.03,
        beta=beta,
        psi=0.2,
        kappa=1.85,
        epsilon=0.06,
        epsilon_child=0.2,
        nu=0.001,
        omega=0.019,
        duration=3.0,
        untreated_cfr=0.4,
        treatment_cfr=0.03,
        detection_logitK=0.0,
        detection_trend=0.0,
        detection_or_0_4=1.0,
        detection_or_5_14=1.0,
    )

    return extended ? merge(
        Base.structdiff(
            base,
            (
                duration=base.duration,
                untreated_cfr=base.untreated_cfr,
            ),
        ),
        TB.extended_fixed_parameters(),
        (
            duration_sp=1.6,
            duration_sn=5.35,
            duration_ep=3.0,
            untreated_cfr_sp=0.7,
            untreated_cfr_sn=0.2,
            untreated_cfr_ep=0.2,
        ),
    ) : base
end

function initialise(model, demography, beta; extended=false)
    params = TB.make_parameters(
        model, draw(beta; extended)
    )

    u0 = TB.initial_state(
        model, params, demography;
        year=FIRST_YEAR,
    )

    return params, u0
end

function living_population(u, layout)
    stride = TB.nstates(layout)

    return [
        sum(
            u[TB.state_index(layout, a, c)]
            for c in 1:layout.nepi
        )
        for a in 1:layout.nages
    ]
end

function boundary_state(sol, year)
    indices = findall(
        t -> abs(t - year) <= 1e-7,
        sol.t,
    )

    length(indices) >= 2 ||
        error("Missing pre/post demographic states at $year")

    return sol.u[first(indices)], sol.u[last(indices)]
end

end # module


@testset "End-to-end 96-age TB models" begin
    TB = TBKiribatiJulia
    F = EndToEndFixture

    demography = F.demographic_scenario()

    for (name, model, extended) in (
        ("Dodd", F.dodd_model(), false),
        ("DoddExtended", F.extended_model(), true),
    )
        @testset "$name" begin
            layout = TB.state_layout(model)

            @test layout.nages == 96
            @test layout.nepi == (extended ? 13 : 6)
            @test layout.ncum == (extended ? 25 : 10)

            params, u0 = F.initialise(
                model, demography, 3.0;
                extended,
            )

            @test length(u0) ==
                TB.state_dimension(layout)

            @test all(isfinite, u0)

            @test all(x -> x >= 0, u0)

            initial_population = F.living_population(
                u0, layout
            )

            target = TB.get_population(
                demography, 2023
            )

            @test initial_population ≈ target rtol=1e-10

            # -----------------------------------------
            # Prepared full-model simulation
            # -----------------------------------------

            setup = TB.prepare_simulation(
                model,
                demography;
                start_year=2023,
                end_year=2025,
                abstol=1e-8,
                reltol=1e-7,
            )

            sol = TB.simulate(setup, params; u0)

            @test successful_retcode(sol)
            @test sol.t[1] ≈ 2023.0
            @test sol.t[end] ≈ 2025.0

            @test all(
                u -> all(isfinite, u),
                sol.u,
            )

            # -----------------------------------------
            # Demographic boundary accounting
            # -----------------------------------------

            for year in (2024, 2025)
                before, after =
                    F.boundary_state(sol, year)

                living = F.living_population(
                    after, layout
                )

                expected = TB.get_population(
                    demography, year
                )

                @test living ≈ expected rtol=1e-9

                # Annual demographic updates must
                # leave cumulative counters unchanged.
                for a in 1:layout.nages
                    for c in (
                            layout.nepi+1
                        ):TB.nstates(layout)
                        i = TB.state_index(
                            layout, a, c
                        )

                        @test after[i] ≈ before[i] atol=1e-8
                    end
                end
            end

            # -----------------------------------------
            # Epidemiological output identities
            # -----------------------------------------

            out = TB.model_outputs(
                model,
                sol;
                variables=(
                    :disease_onsets,
                    :treatment_initiations,
                    :tb_deaths,
                    :untreated_active_prevalence,
                    :latent_prevalence,
                    :treatment_prevalence,
                ),
                years=2023:2024,
                age_breaks=[0, 5, 15, 25, 35,
                    45, 55, 65, 96],
            )

            for variable in (
                :disease_onsets,
                :treatment_initiations,
                :tb_deaths,
                :untreated_active_prevalence,
                :latent_prevalence,
                :treatment_prevalence,
            )
                values = getproperty(out, variable)

                @test size(values) == (8, 2)
                @test all(isfinite, values)
            end

            # Directly modelled events should be
            # nonnegative to numerical tolerance.
            for variable in (
                :disease_onsets,
                :treatment_initiations,
                :tb_deaths,
            )
                values = getproperty(out, variable)

                @test minimum(values) >= -1e-6
            end

            # -----------------------------------------
            # Differentiation through complete solver
            # -----------------------------------------

            function prediction(beta)
                p, initial = F.initialise(
                    model,
                    demography,
                    beta;
                    extended,
                )

                result = TB.simulate(
                    setup, p; u0=initial
                )

                @assert successful_retcode(result)

                notifications = TB.model_output(
                    model,
                    result,
                    :treatment_initiations;
                    years=2023:2024,
                )

                return sum(notifications)
            end

            beta0 = 3.0
            h = 1e-4

            grad_ad = ForwardDiff.derivative(
                prediction, beta0
            )

            grad_fd = (
                prediction(beta0 + h) -
                    prediction(beta0 - h)
            ) / (2h)

            @test isfinite(grad_ad)
            @test isfinite(grad_fd)

            @test isapprox(
                grad_ad, grad_fd;
                rtol=5e-3,
                atol=1e-3,
            )
        end
    end
end
