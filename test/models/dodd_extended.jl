
using Test
using LinearAlgebra
using TBKiribatiJulia

const TB = TBKiribatiJulia

@testset "Independent DoddExtended model" begin
    n = 20
    C = ones(n, n)
    P = repeat(
        reshape([0.5, 0.3, 0.2], 1, 3),
        n, 1
    )

    model = TB.DoddExtendedTBModel(C, P)

    @test TB.state_layout(model).nepi == 13
    @test TB.state_layout(model).ncum == 25
    @test TB.nstates(TB.state_layout(model)) == 38
    @test size(model.contact) == (n, n)
    @test sum(model.contact) / n ≈ 1.0

    @test !hasproperty(model, :reference)
    @test !hasproperty(model, :base)

    draw = (
        Lambda0=0.01,
        beta=2.0,
        psi=0.2,
        kappa=1.5,
        epsilon=0.1,
        epsilon_child=0.2,
        nu=0.02,
        omega=0.05,
        duration_sp=1.6,
        duration_sn=5.35,
        duration_ep=3.0,
        untreated_cfr_sp=0.70,
        untreated_cfr_sn=0.20,
        untreated_cfr_ep=0.20,
        treatment_cfr=0.05,
        detection_logitK=0.0,
        detection_trend=0.01,
        detection_or_0_4=1.2,
        detection_or_5_14=1.1,
        smear_negative_infectiousness=0.35,
        smear_negative_detection=1.2,
        extrapulmonary_detection=1.0,
    )

    p = TB.make_parameters(model, draw)

    for (k, (d, c)) in enumerate((
        (1.6, 0.7),
        (5.35, 0.2),
        (3.0, 0.2),
    ))
        @test p.self_cure[16, k] ≈ (1 - c) / d
        @test p.tb_mortality[16, k] ≈ c / d
    end

    for k in 1:3
        @test p.self_cure[1, k] +
            p.tb_mortality[1, k] ≈ 2.0
    end

    # Detection hazard scaling is independent of
    # presentation-specific untreated durations.
    @test p.detection_multiplier[16] ≈ 1 / 3

    @test p isa TB.DoddExtendedTBParameters
    @test !hasproperty(p, :base)

    @test p.beta == 2.0
    @test p.psi == 0.2
    @test p.epsilon[1] == 0.2
    @test p.epsilon[16] == 0.1
    @test size(p.self_cure) == (n, 3)
    @test size(p.tb_mortality) == (n, 3)

    durations = (1.6, 5.35, 3.0)
    cfrs = (0.7, 0.2, 0.2)

    for a in 16:n, k in 1:3
        d = durations[k]
        c = cfrs[k]

        @test p.self_cure[a, k] ≈ (1 - c) / d
        @test p.tb_mortality[a, k] ≈ c / d
        @test p.self_cure[a, k] +
            p.tb_mortality[a, k] ≈ 1 / d
    end

    @test p.relative_infectiousness ==
        (1.0, 0.35, 0.0)

    @test p.relative_detection ==
        (1.0, 1.2, 1.0)

    # Synthetic state with active TB in multiple
    # presentation and treatment-history compartments.
    layout = TB.state_layout(model)
    stride = TB.nstates(layout)
    u = zeros(stride * n)

    for a in 1:n
        b = (a-1)*stride
        u[b+1] = 900.0
        u[b+4] = 100.0
        u[b+6] = 5.0
        u[b+7] = 3.0
        u[b+9] = 2.0
        u[b+12] = 1.0
    end

    du = fill(NaN, length(u))

    ctx = TB.SimulationContext(
        model, p, nothing,
        TB.TBWorkspace(n),
    )

    TB.tb_rhs!(du, u, ctx, 2023.0)

    bad = findall(isnan, du)

    println("\nUnassigned derivatives:")

    for i in bad
        age = div(i - 1, stride) + 1
        state = mod(i - 1, stride) + 1

        println("Age index: $age, state index: $state")
    end

    @test all(isfinite, du)

    # Continuous epidemiological dynamics conserve
    # living population except for TB mortality.
    D = reshape(du, stride, n)

    for a in 1:n
        living = sum(@view D[1:13, a])
        deaths = D[21, a] + D[22, a]

        @test living + deaths ≈ 0.0 atol=1e-9
    end

    # Reinfection can be disabled without a psi draw.
    reduced = TB.DoddExtendedTBModel(
        C, P; reinfection=false
    )

    reduced_draw = Base.structdiff(
        draw, (; psi=draw.psi)
    )

    p_reduced = TB.make_parameters(
        reduced, reduced_draw
    )

    @test iszero(p_reduced.psi)
    @test :psi ∉ keys(TB.priors(reduced))
end
