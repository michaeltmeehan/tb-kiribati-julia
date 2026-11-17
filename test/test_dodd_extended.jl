
using Test
using LinearAlgebra
using Distributions: median
using TBKiribatiJulia

const DE = TBKiribatiJulia.DoddExtendedStates

@testset "Dodd extended model" begin

    nages = 96
    contact = Matrix{Float64}(I, nages, nages)

    # Synthetic presentation, independent of WHO data.
    presentation = fill(1 / 3, nages, 3)

    model = DoddExtendedTBModel(
        contact,
        presentation,
    )

    layout = state_layout(model)

    @test layout.nepi == 13
    @test layout.ncum == 16
    @test nstates(layout) == 29

    # Additional parameters and priors.
    prior = priors(model)

    @test haskey(prior, :smear_negative_infectiousness)
    @test haskey(prior, :smear_negative_detection)
    @test haskey(prior, :extrapulmonary_detection)

    draw = merge(
        map(median, prior),
        extended_fixed_parameters(),
    )

    params = make_parameters(model, draw)

    @test params.relative_infectiousness ==
        (1.0, 0.35, 0.0)

    @test params.relative_detection ==
        (1.0, 1.0, 1.0)

    # --------------------------------------------------------
    # Population conservation in the continuous dynamics
    # --------------------------------------------------------

    nstate = nstates(layout)
    u = zeros(nages * nstate)

    for a in 1:nages
        b = (a - 1) * nstate
        u[b+DE.S] = 100.0
    end

    # Populate several disease compartments.
    b = (20 - 1) * nstate

    u[b+DE.S] -= 20.0
    u[b+DE.E0] = 10.0
    u[b+DE.L0] = 5.0
    u[b+DE.L1] = 1.0
    u[b+DE.I0SP] = 1.0
    u[b+DE.I1SN] = 1.0
    u[b+DE.T] = 1.0
    u[b+DE.R] = 1.0

    workspace = TBKiribatiJulia.TBWorkspace(nages)

    ctx = TBKiribatiJulia.SimulationContext(
        model, params, nothing, workspace
    )

    du = similar(u)

    TBKiribatiJulia.tb_rhs!(
        du, u, ctx, 2024.0
    )

    for a in 1:nages
        b = (a - 1) * nstate

        population_change = sum(
            du[(b+1):(b+layout.nepi)]
        )

        tb_deaths =
            du[b+DE.CumUntreatedTBDeaths] +
                du[b+DE.CumTreatmentTBDeaths]

        @test isapprox(
            population_change,
            -tb_deaths;
            atol=1e-9,
        )

        notification_total =
            du[b+DE.CumNewSP] +
                du[b+DE.CumNewSN] +
                du[b+DE.CumNewEP] +
                du[b+DE.CumRecurrentSP] +
                du[b+DE.CumRecurrentSN] +
                du[b+DE.CumRecurrentEP]

        @test isapprox(
            notification_total,
            du[b+DE.CumTreatmentInitiation];
            atol=1e-9,
        )
    end

    # --------------------------------------------------------
    # Reduced model: reinfection absent
    # --------------------------------------------------------

    reduced = DoddExtendedTBModel(
        contact,
        presentation;
        reinfection=false,
    )

    reduced_prior = priors(reduced)

    @test !haskey(reduced_prior, :psi)

    reduced_draw = merge(
        map(median, reduced_prior),
        extended_fixed_parameters(),
    )

    reduced_params = make_parameters(
        reduced, reduced_draw
    )

    @test reduced_params.base.psi == 0.0

    reduced_ctx = TBKiribatiJulia.SimulationContext(
        reduced,
        reduced_params,
        nothing,
        TBKiribatiJulia.TBWorkspace(nages),
    )

    du_reduced = similar(u)

    TBKiribatiJulia.tb_rhs!(
        du_reduced, u, reduced_ctx, 2024.0
    )

    # E1 is initially empty: no inflow without reinfection.
    @test du_reduced[b+DE.E1] ≈ 0.0

end
