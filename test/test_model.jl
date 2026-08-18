using Test

function zero_transition_params(; beta = 0.0)
    p = make_parameters(default_contact_matrix();
        beta = beta,
        containment_child = 0.0,
        containment_adult = 0.0,
        clearance_rate = 0.0,
        breakdown_rate = 0.0,
        prop_infectious = 0.0,
        progression_child = 0.0,
        progression_5_14 = 0.0,
        progression_15_64 = 0.0,
        progression_65_plus = 0.0,
        clinical_progression_rate = 0.0,
        clinical_regression_rate = 0.0,
        infectiousness_gain_rate = 0.0,
        infectiousness_loss_rate = 0.0,
        self_recovery_rate = 0.0,
        detection_rate = 0.0,
        rel_detection_subclin = 0.0,
        tx_recovery_rate = 0.0,
        tx_relapse_rate = 0.0,
        tx_death_rate = 0.0,
        disease_mortality_clin_lowinf = 0.0,
        disease_mortality_clin_inf = 0.0)
    fill!(p.containment_rate, 0.0)
    fill!(p.progression_rate, 0.0)
    return p
end

function age_index(age::Integer)
    age + 1
end

function blank_state()
    zeros(Float64, NSTATE * NAGE)
end

@testset "Force of infection" begin
    p = zero_transition_params(; beta = 1e-3)
    fill!(p.contact, 0.0)
    p.contact[1, age_index(16)] = 2.0
    p.contact[1, age_index(17)] = 4.0
    p.contact[2, age_index(16)] = 1.0
    p.contact[2, age_index(17)] = 3.0
    u = blank_state()
    U = reshape(u, NSTATE, NAGE)

    U[SubClinLow, age_index(16)] = 10.0
    U[SubClinInf, age_index(16)] = 20.0
    U[ClinLow, age_index(16)] = 30.0
    U[ClinInf, age_index(16)] = 40.0
    U[MtbNaive, age_index(16)] = 0.0
    U[SubClinLow, age_index(17)] = 0.0
    U[SubClinInf, age_index(17)] = 0.0
    U[ClinLow, age_index(17)] = 0.0
    U[ClinInf, age_index(17)] = 50.0
    U[MtbNaive, age_index(16)] = 100.0
    U[MtbNaive, age_index(17)] = 100.0

    λ = zeros(Float64, NAGE)
    compute_force_of_infection!(λ, u, p)

    q16 = (0.2 * 10.0 + 0.5 * 20.0 + 0.4 * 30.0 + 1.0 * 40.0) / 200.0
    q17 = (0.2 * 0.0 + 0.5 * 0.0 + 0.4 * 0.0 + 1.0 * 50.0) / 150.0
    @test isapprox(λ[1], 1e-3 * (2.0 * q16 + 4.0 * q17); atol = 1e-12)
    @test isapprox(λ[2], 1e-3 * (1.0 * q16 + 3.0 * q17); atol = 1e-12)

    fill!(u, 0.0)
    U = reshape(u, NSTATE, NAGE)
    U[SubClinLow, age_index(10)] = 25.0
    U[ClinInf, age_index(10)] = 25.0
    U[MtbNaive, age_index(20)] = 100.0
    fill!(λ, 0.0)
    compute_force_of_infection!(λ, u, p)
    @test all(iszero, λ)
end

@testset "No infection without infectious individuals" begin
    p = zero_transition_params(; beta = 1e-3)
    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(20)] = 100.0
    U[Contained, age_index(20)] = 50.0
    U[Cleared, age_index(20)] = 25.0
    U[Recovered, age_index(20)] = 25.0

    λ = zeros(Float64, NAGE)
    compute_force_of_infection!(λ, u, p)
    @test all(iszero, λ)

    du = similar(u)
    tb_rhs!(du, u, p, 0.0)
    @test all(iszero, du)
end

@testset "Childhood infectiousness" begin
    p = zero_transition_params(; beta = 1e-3)
    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[SubClinLow, age_index(10)] = 20.0
    U[SubClinInf, age_index(10)] = 20.0
    U[ClinLow, age_index(10)] = 20.0
    U[ClinInf, age_index(10)] = 20.0
    U[MtbNaive, age_index(20)] = 100.0

    λ = zeros(Float64, NAGE)
    compute_force_of_infection!(λ, u, p)
    @test all(iszero, λ)
end

@testset "Population conservation" begin
    p = make_parameters(default_contact_matrix();
        beta = 1e-3,
        disease_mortality_clin_lowinf = 0.0,
        disease_mortality_clin_inf = 0.0,
        tx_death_rate = 0.0)
    u = initial_state(default_population())
    U = reshape(u, NSTATE, NAGE)
    U[Contained, age_index(20)] = 50.0
    U[Incipient, age_index(20)] = 40.0
    U[SubClinLow, age_index(20)] = 30.0
    U[SubClinInf, age_index(20)] = 20.0
    U[ClinLow, age_index(20)] = 10.0
    U[ClinInf, age_index(20)] = 5.0
    U[Treatment, age_index(20)] = 2.0
    U[MtbNaive, age_index(20)] = max(0.0, 1000.0 - sum(@view U[1:10, age_index(20)]))

    du = similar(u)
    tb_rhs!(du, u, p, 0.0)
    DU = reshape(du, NSTATE, NAGE)
    @test isapprox(sum(DU[1:NEPI, :]), 0.0; atol = 1e-10, rtol = 1e-10)
end

@testset "Infection conservation" begin
    p = zero_transition_params(; beta = 1e-3)
    fill!(p.containment_rate, 0.0)
    fill!(p.progression_rate, 0.0)

    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(20)] = 90.0
    U[Contained, age_index(20)] = 40.0
    U[Cleared, age_index(20)] = 30.0
    U[Recovered, age_index(20)] = 20.0
    U[ClinInf, age_index(20)] = 100.0
    U[MtbNaive, age_index(20)] += 10.0

    du = similar(u)
    tb_rhs!(du, u, p, 0.0)
    DU = reshape(du, NSTATE, NAGE)
    infection_loss = DU[MtbNaive, age_index(20)] + DU[Contained, age_index(20)] + DU[Cleared, age_index(20)] + DU[Recovered, age_index(20)]
    @test isapprox(infection_loss + DU[Incipient, age_index(20)], 0.0; atol = 1e-10, rtol = 1e-10)
end

@testset "Ordinary transition conservation" begin
    age = age_index(20)

    transition_checks = [
        (:incipient_to_contained, Incipient, Contained, :containment_rate, 1.0, (p, u) -> nothing),
        (:contained_to_cleared, Contained, Cleared, :clearance_rate, 1.0, (p, u) -> nothing),
        (:contained_to_incipient, Contained, Incipient, :breakdown_rate, 1.0, (p, u) -> nothing),
        (:incipient_to_sub_low, Incipient, SubClinLow, :progression_rate, 1.0, (p, u) -> (p.prop_infectious = 0.0; nothing)),
        (:incipient_to_sub_inf, Incipient, SubClinInf, :progression_rate, 1.0, (p, u) -> (p.prop_infectious = 1.0; nothing)),
        (:sub_low_to_clin_low, SubClinLow, ClinLow, :clinical_progression_rate, 1.0, (p, u) -> nothing),
        (:sub_inf_to_clin_inf, SubClinInf, ClinInf, :clinical_progression_rate, 1.0, (p, u) -> nothing),
        (:clin_low_to_sub_inf, ClinLow, SubClinInf, :clinical_regression_rate, 1.0, (p, u) -> nothing),
        (:clin_inf_to_sub_inf, ClinInf, SubClinInf, :clinical_regression_rate, 1.0, (p, u) -> nothing),
        (:sub_low_to_sub_inf, SubClinLow, SubClinInf, :infectiousness_gain_rate, 1.0, (p, u) -> nothing),
        (:clin_low_to_clin_inf, ClinLow, ClinInf, :infectiousness_gain_rate, 1.0, (p, u) -> nothing),
        (:sub_inf_to_sub_low, SubClinInf, SubClinLow, :infectiousness_loss_rate, 1.0, (p, u) -> nothing),
        (:clin_inf_to_clin_low, ClinInf, ClinLow, :infectiousness_loss_rate, 1.0, (p, u) -> nothing),
        (:sub_low_to_recovered, SubClinLow, Recovered, :self_recovery_rate, 1.0, (p, u) -> nothing),
        (:sub_inf_to_recovered, SubClinInf, Recovered, :self_recovery_rate, 1.0, (p, u) -> nothing),
        (:sub_low_to_treatment, SubClinLow, Treatment, :detection_rate, 1.0, (p, u) -> (p.rel_detection_subclin = 1.0; nothing)),
        (:sub_inf_to_treatment, SubClinInf, Treatment, :detection_rate, 1.0, (p, u) -> (p.rel_detection_subclin = 1.0; nothing)),
        (:clin_low_to_treatment, ClinLow, Treatment, :detection_rate, 1.0, (p, u) -> nothing),
        (:clin_inf_to_treatment, ClinInf, Treatment, :detection_rate, 1.0, (p, u) -> nothing),
        (:treatment_to_recovered, Treatment, Recovered, :tx_recovery_rate, 1.0, (p, u) -> nothing),
        (:treatment_to_relapse, Treatment, SubClinLow, :tx_relapse_rate, 1.0, (p, u) -> nothing),
    ]

    for (_, src, dst, rate_name, rate_value, tweak) in transition_checks
        p = zero_transition_params()
        u = blank_state()
        U = reshape(u, NSTATE, NAGE)
        U[src, age] = 10.0
        U[MtbNaive, age] = 0.0
        if rate_name === :containment_rate || rate_name === :progression_rate
            fill!(getfield(p, rate_name), 0.0)
            getfield(p, rate_name)[age] = rate_value
        else
            setfield!(p, rate_name, rate_value)
        end
        tweak(p, u)
        du = similar(u)
        tb_rhs!(du, u, p, 0.0)
        DU = reshape(du, NSTATE, NAGE)
        gain = DU[dst, age]
        loss = DU[src, age]
        @test isapprox(loss + gain, 0.0; atol = 1e-12, rtol = 1e-12)
    end
end

@testset "Cumulative flows" begin
    p = zero_transition_params(; beta = 1e-3)
    fill!(p.containment_rate, 0.0)
    fill!(p.progression_rate, 0.0)
    p.containment_rate[age_index(20)] = 2.0
    p.progression_rate[age_index(20)] = 3.0
    p.prop_infectious = 0.25
    p.clinical_progression_rate = 4.0
    p.clinical_regression_rate = 5.0
    p.infectiousness_gain_rate = 6.0
    p.infectiousness_loss_rate = 7.0
    p.self_recovery_rate = 8.0
    p.detection_rate = 9.0
    p.rel_detection_subclin = 0.5
    p.tx_recovery_rate = 10.0
    p.tx_relapse_rate = 11.0

    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    a = age_index(20)
    U[MtbNaive, a] = 100.0
    U[Contained, a] = 20.0
    U[Cleared, a] = 30.0
    U[Recovered, a] = 40.0
    U[Incipient, a] = 50.0
    U[SubClinLow, a] = 60.0
    U[SubClinInf, a] = 70.0
    U[ClinLow, a] = 80.0
    U[ClinInf, a] = 90.0
    U[Treatment, a] = 11.0
    U[ClinInf, a] += 10.0

    du = similar(u)
    tb_rhs!(du, u, p, 0.0)
    DU = reshape(du, NSTATE, NAGE)

    λ = zeros(Float64, NAGE)
    compute_force_of_infection!(λ, u, p)
    inf_naive = λ[a] * p.susceptibility_naive[a] * U[MtbNaive, a]
    inf_contained = λ[a] * p.susceptibility_contained[a] * U[Contained, a]
    inf_cleared = λ[a] * p.susceptibility_cleared[a] * U[Cleared, a]
    inf_recovered = λ[a] * p.susceptibility_recovered[a] * U[Recovered, a]
    to_sub_low = (1 - p.prop_infectious) * p.progression_rate[a] * U[Incipient, a]
    to_sub_inf = p.prop_infectious * p.progression_rate[a] * U[Incipient, a]
    to_tx = p.rel_detection_subclin * p.detection_rate * U[SubClinLow, a] +
            p.rel_detection_subclin * p.detection_rate * U[SubClinInf, a] +
            p.detection_rate * U[ClinLow, a] +
            p.detection_rate * U[ClinInf, a]
    tx_rec = p.tx_recovery_rate * U[Treatment, a]
    tx_rel = p.tx_relapse_rate * U[Treatment, a]

    @test isapprox(DU[CumInfectionsOther, a], inf_naive + inf_cleared + inf_recovered; atol = 1e-12, rtol = 1e-12)
    @test isapprox(DU[CumInfectionsContained, a], inf_contained; atol = 1e-12, rtol = 1e-12)
    @test isapprox(DU[CumProgressionToActiveTB, a], to_sub_low + to_sub_inf; atol = 1e-12, rtol = 1e-12)
    @test isapprox(DU[CumTreatmentInitiation, a], to_tx; atol = 1e-12, rtol = 1e-12)
    @test isapprox(DU[CumTreatmentCompletion, a], tx_rec; atol = 1e-12, rtol = 1e-12)
    @test isapprox(DU[CumRelapseTB, a], tx_rel; atol = 1e-12, rtol = 1e-12)
end

@testset "Numerical sanity" begin
    sol = simulate_demo(tspan = (2025.0, 2025.5), saveat = 0.25)
    @test occursin("Success", string(sol.retcode))

    for state in sol.u
        @test all(isfinite, state)
        epi = reshape(state, NSTATE, NAGE)
        @test minimum(epi[1:NEPI, :]) ≥ -1e-8
    end

    cum_idxs = CumInfectionsOther:CumRelapseTB
    for idx in cum_idxs
        series = [reshape(state, NSTATE, NAGE)[idx, 1] for state in sol.u]
        @test all(diff(series) .>= -1e-10)
    end
end
