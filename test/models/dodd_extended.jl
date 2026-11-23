
using Test
using LinearAlgebra
using TBKiribatiJulia

const TB = TBKiribatiJulia


struct TestDoddExtendedDemography <: TB.AbstractDemography
    population::Vector{Float64}
end

TB.get_population(
    d::TestDoddExtendedDemography,
    year::Real,
) = d.population


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
        early_duration=2.0,
        fast_risk_0=0.45,
        fast_risk_1=0.18,
        fast_risk_2_4=0.055,
        fast_risk_5_9=0.02,
        fast_risk_10_14=0.10,
        fast_risk_15_plus=0.05,
        nu=0.02,
        omega=0.05,
        duration_ep=3.0,
        mortality_sp=0.389,
        self_cure_sp=0.231,
        mortality_sn=0.025,
        self_cure_sn=0.130,
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

    @testset "Pathway-specific presentation" begin
        @test size(p.presentation_fast) == (n, 3)
        @test size(p.presentation_slow) == (n, 3)
        @test size(p.presentation_relapse) == (n, 3)

        @test p.presentation_fast ≈ P
        @test p.presentation_slow ≈ P
        @test p.presentation_relapse ≈ P

        for M in (
            p.presentation_fast,
            p.presentation_slow,
            p.presentation_relapse,
        )
            @test all(isapprox.(
                vec(sum(M; dims=2)), 1.0
            ))
            @test all((0 .<= M) .& (M .<= 1))
        end
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
    @test p.epsilon[1] ≈ 0.45 / 2.0
    @test p.epsilon[16] ≈ 0.05 / 2.0

    @test p.kappa[1] ≈ 0.55 / 2.0
    @test p.kappa[16] ≈ 0.95 / 2.0
    @test size(p.self_cure) == (n, 3)
    @test size(p.tb_mortality) == (n, 3)

    @testset "Presentation-specific natural history" begin
        expected_gamma = [0.231, 0.130, 0.8 / 3.0]
        expected_mu = [0.389, 0.025, 0.2 / 3.0]

        # Adult hazards reproduce the input parameters.
        for a in 16:n, k in 1:3
            @test p.self_cure[a, k] ≈ expected_gamma[k]
            @test p.tb_mortality[a, k] ≈ expected_mu[k]
        end

        # Derived durations and fatality probabilities.
        for k in 1:3
            gamma = p.self_cure[16, k]
            mu = p.tb_mortality[16, k]

            duration = 1 / (gamma + mu)
            cfr = mu / (gamma + mu)

            @test duration ≈
                1 / (expected_gamma[k] + expected_mu[k])

            @test cfr ≈
                expected_mu[k] /
                (expected_gamma[k] + expected_mu[k])
        end

        # Existing childhood duration assumption.
        for a in 1:15, k in 1:3
            @test p.self_cure[a, k] +
                p.tb_mortality[a, k] ≈ 2.0
        end

        # Check that the new prior fields exist.
        prior = TB.priors(model)

        for name in (
            :mortality_sp, :self_cure_sp,
            :mortality_sn, :self_cure_sn,
        )
            @test name in keys(prior)
        end

        @test :duration_sp ∉ keys(prior)
        @test :duration_sn ∉ keys(prior)
        @test :untreated_cfr_sp ∉ keys(prior)
        @test :untreated_cfr_sn ∉ keys(prior)
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

    @testset "Age-specific fast progression" begin
        risks = [0.45, 0.18, 0.055, 0.02, 0.10, 0.05]

        r = TBKiribatiJulia.expand_fast_progression_risks(risks, 96)

        epsilon, kappa = TBKiribatiJulia.fast_progression_hazards(r, 2.0)

        @test length(epsilon) == 96
        @test length(kappa) == 96

        @test epsilon ./ (epsilon .+ kappa) ≈ r

        @test all(isapprox.(
            epsilon .+ kappa, 0.5
        ))

        @test r[1] == risks[1]   # Age 0
        @test r[2] == risks[2]   # Age 1
        @test r[6] == risks[4]   # Age 5
        @test r[16] == risks[6]  # Age 15
    end

    @testset "DoddExtended initialisation" begin
        # Synthetic population matching the model's 20 ages.
        population = fill(1000.0, n)
        demography = TestDoddExtendedDemography(population)

        u0 = TB.initial_state(
            model, p, demography;
            year=2023.0,
        )

        layout = TB.state_layout(model)
        nages = layout.nages
        stride = TB.nstates(layout)

        U = reshape(u0, stride, nages)
        DE = TB.DoddExtendedStates

        @test length(u0) == stride * nages
        @test all(isfinite, u0)
        @test all(u0 .>= -1e-10)

        # Population conservation at each age.
        @test vec(sum(U[1:layout.nepi, :]; dims=1)) ≈
            population

        # No recurrence or reinfection at initialisation.
        @test all(U[DE.E1, :] .== 0)
        @test all(U[DE.L1, :] .== 0)
        @test all(U[DE.I1SP, :] .== 0)
        @test all(U[DE.I1SN, :] .== 0)
        @test all(U[DE.I1EP, :] .== 0)

        # Cumulative event counters start at zero.
        @test all(U[(layout.nepi+1):end, :] .== 0)

        # At birth, most individuals should be susceptible.
        @test U[DE.S, 1] > U[DE.L0, 1]

        # Older ages accumulate infection history.
        @test U[DE.S, end] < U[DE.S, 1]
    end

    @testset "Pathway-specific incidence allocation" begin
        # Distinct presentation distributions for each pathway.
        pf = repeat(reshape([0.8, 0.1, 0.1], 1, 3), n, 1)
        ps = repeat(reshape([0.2, 0.7, 0.1], 1, 3), n, 1)
        pr = repeat(reshape([0.1, 0.2, 0.7], 1, 3), n, 1)

        # Replace the matrices in an otherwise identical
        # parameter object.
        fields = fieldnames(typeof(p))
        values = map(fields) do f
            f === :presentation_fast ? pf :
            f === :presentation_slow ? ps :
            f === :presentation_relapse ? pr :
            getfield(p, f)
        end

        pp = TB.DoddExtendedTBParameters(values...)

        # One age class with known compartment occupancies.
        a = 16
        b = (a - 1) * stride

        v = zeros(length(u))
        v[b+TB.DoddExtendedStates.E0] = 10.0
        v[b+TB.DoddExtendedStates.L0] = 20.0
        v[b+TB.DoddExtendedStates.E1] = 5.0
        v[b+TB.DoddExtendedStates.L1] = 8.0
        v[b+TB.DoddExtendedStates.R] = 4.0

        dv = zeros(length(v))

        test_ctx = TB.SimulationContext(
            model, pp, nothing, TB.TBWorkspace(n)
        )

        TB.tb_rhs!(dv, v, test_ctx, 2023.0)

        DE = TB.DoddExtendedStates

        fast = pp.epsilon[a] * (10.0 + 5.0)
        slow = pp.nu[a] * (20.0 + 8.0)
        relapse = pp.omega[a] * 4.0

        expected = [
            pf[a, k] * fast +
                ps[a, k] * slow +
                pr[a, k] * relapse
            for k in 1:3
        ]

        observed = [
            dv[b+DE.CumIncidenceSP+k-1]
            for k in 1:3
        ]

        @test observed ≈ expected
        @test sum(observed) ≈ fast + slow + relapse

        # Verify the individual primary/recurrent
        # compartment inflows as well.
        for k in 1:3
            expected_0 =
                pf[a, k] * pp.epsilon[a] * 10.0 +
                    ps[a, k] * pp.nu[a] * 20.0

            expected_1 =
                pf[a, k] * pp.epsilon[a] * 5.0 +
                    ps[a, k] * pp.nu[a] * 8.0 +
                    pr[a, k] * relapse

            @test dv[b+DE.I0SP+k-1] ≈ expected_0
            @test dv[b+DE.I1SP+k-1] ≈ expected_1
        end
    end

    @testset "Fast-progression EP presentation modifier" begin
        p_alt = TB.make_parameters(
            model, draw;
            fast_ep_odds_under5=2.0,
        )

        # Baseline has EP probability 0.2.
        # Doubling its odds gives 1/3.
        @test p_alt.presentation_fast[1, 3] ≈ 1 / 3

        # SP/SN relative odds are preserved.
        @test (
            p_alt.presentation_fast[1, 1] /
                p_alt.presentation_fast[1, 2]
        ) ≈ P[1, 1] / P[1, 2]

        # Ages 5+ are unaffected.
        @test p_alt.presentation_fast[6:end, :] ≈
            P[6:end, :]

        # Slow progression and relapse are unaffected.
        @test p_alt.presentation_slow ≈ P
        @test p_alt.presentation_relapse ≈ P

        # Probabilities remain normalised.
        @test all(isapprox.(
            vec(sum(p_alt.presentation_fast; dims=2)),
            1.0
        ))

        # The baseline parameterisation is preserved.
        @test TB.make_parameters(
            model, draw
        ).presentation_fast ≈ P

        @test_throws ArgumentError TB.make_parameters(
            model, draw;
            fast_ep_odds_under5=-1.0,
        )
    end

    @testset "Presentation-specific ascertainment" begin
        # Analytical competing-risk example.
        P_test = reshape([0.5, 0.3, 0.2], 1, 3)

        # Diagnosis probabilities 1/2, 1/3 and 1/4.
        A_test = reshape([0.5, 1/3, 0.25], 1, 3)

        Q = TB.expected_notification_presentation(
            P_test, A_test
        )

        @test Q[1, :] ≈ [0.625, 0.25, 0.125]
        @test sum(Q) ≈ 1.0

        # Equal ascertainment preserves the distribution.
        Q_equal = TB.expected_notification_presentation(
            P_test, fill(0.5, 1, 3)
        )

        @test Q_equal ≈ P_test

        # Test against actual DoddExtended hazards.
        A = TB.presentation_ascertainment(p, 2023.0)

        println("\nPresentation-specific ascertainment (2023)")
        println("Age    SP       SN       EP")

        for age in (0, 5, 10, 15, 19)
            a = age + 1

            println(
                rpad(age, 7),
                join(
                    lpad.(string.(round.(A[a, :]; digits=3)), 9),
                    ""
                )
            )
        end

        @test size(A) == (n, 3)
        @test all((0 .<= A) .& (A .<= 1))

        Q_model = TB.expected_notification_presentation(
            p.presentation_fast, A
        )

        @test size(Q_model) == (n, 3)
        @test all(isapprox.(
            vec(sum(Q_model; dims=2)), 1.0
        ))
    end

end
