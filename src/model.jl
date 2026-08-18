using LinearAlgebra: I, mul!
using ADTypes: AutoFiniteDiff
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Rosenbrock23

const NAGE = 96
const NEPI = 10
const NCUM = 6
const NSTATE = NEPI + NCUM

const MtbNaive = 1
const Contained = 2
const Cleared = 3
const Recovered = 4
const Incipient = 5
const SubClinLow = 6
const SubClinInf = 7
const ClinLow = 8
const ClinInf = 9
const Treatment = 10

const CumInfectionsOther = 11
const CumInfectionsContained = 12
const CumProgressionToActiveTB = 13
const CumTreatmentInitiation = 14
const CumTreatmentCompletion = 15
const CumRelapseTB = 16

export NAGE, NEPI, NCUM, NSTATE
export MtbNaive, Contained, Cleared, Recovered, Incipient, SubClinLow, SubClinInf, ClinLow, ClinInf, Treatment
export CumInfectionsOther, CumInfectionsContained, CumProgressionToActiveTB, CumTreatmentInitiation, CumTreatmentCompletion, CumRelapseTB
export TBParams, DemographicSchedule, make_parameters, make_default_parameters, make_demographic_parameters, default_contact_matrix, default_population, initial_state
export compute_force_of_infection!, tb_rhs_epi!, tb_rhs!, apply_demography!, synthetic_demographic_schedule, simulate_demo, simulate_demographic_demo

function compute_force_of_infection!(λ::AbstractVector{<:Real}, u::AbstractVector, p::TBParams)
    length(λ) == NAGE || error("λ must have length 96")
    q = p.tmp_q
    @inbounds for a in 1:NAGE
        age = a - 1
        base = (a - 1) * NSTATE
        if age < 15
            q[a] = 0.0
        else
            total = 0.0
            for c in 1:NEPI
                total += u[base + c]
            end
            if total > 0
                weighted = p.infectiousness_weights[1] * u[base + SubClinLow] +
                           p.infectiousness_weights[2] * u[base + SubClinInf] +
                           p.infectiousness_weights[3] * u[base + ClinLow] +
                           p.infectiousness_weights[4] * u[base + ClinInf]
                q[a] = weighted / total
            else
                q[a] = 0.0
            end
        end
    end
    mul!(λ, p.contact, q)
    @inbounds for a in 1:NAGE
        λ[a] *= p.beta
    end
    return λ
end

function tb_rhs_epi!(du, u, p::TBParams, t)
    λ = p.tmp_foi
    compute_force_of_infection!(λ, u, p)

    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        λa = λ[a]

        naive = u[base + MtbNaive]
        contained = u[base + Contained]
        cleared = u[base + Cleared]
        recovered = u[base + Recovered]
        incipient = u[base + Incipient]
        sublow = u[base + SubClinLow]
        subinf = u[base + SubClinInf]
        clinlow = u[base + ClinLow]
        clininf = u[base + ClinInf]
        tx = u[base + Treatment]

        inf_naive = λa * p.susceptibility_naive[a] * naive
        inf_contained = λa * p.susceptibility_contained[a] * contained
        inf_cleared = λa * p.susceptibility_cleared[a] * cleared
        inf_recovered = λa * p.susceptibility_recovered[a] * recovered

        to_contained = p.containment_rate[a] * incipient
        to_cleared = p.clearance_rate * contained
        to_breakdown = p.breakdown_rate * contained

        to_sub_low = (1 - p.prop_infectious) * p.progression_rate[a] * incipient
        to_sub_inf = p.prop_infectious * p.progression_rate[a] * incipient

        to_clin_low = p.clinical_progression_rate * sublow
        to_clin_inf = p.clinical_progression_rate * subinf
        to_sub_inf_from_clin_low = p.clinical_regression_rate * clinlow
        to_sub_inf_from_clin_inf = p.clinical_regression_rate * clininf
        to_sub_low_from_gain = p.infectiousness_gain_rate * sublow
        to_clin_inf_from_gain = p.infectiousness_gain_rate * clinlow
        to_sub_low_from_loss = p.infectiousness_loss_rate * subinf
        to_clin_low_from_loss = p.infectiousness_loss_rate * clininf

        to_rec_from_sub_low = p.self_recovery_rate * sublow
        to_rec_from_sub_inf = p.self_recovery_rate * subinf

        to_tx_from_sub_low = p.rel_detection_subclin * p.detection_rate * sublow
        to_tx_from_sub_inf = p.rel_detection_subclin * p.detection_rate * subinf
        to_tx_from_clin_low = p.detection_rate * clinlow
        to_tx_from_clin_inf = p.detection_rate * clininf

        tx_to_rec = p.tx_recovery_rate * tx
        tx_to_relapse = p.tx_relapse_rate * tx
        tx_to_death = p.tx_death_rate * tx

        clinlow_death = p.disease_mortality_clin_lowinf * clinlow
        clininf_death = p.disease_mortality_clin_inf * clininf

        du[base + MtbNaive] -= inf_naive

        du[base + Contained] += to_contained - to_cleared - to_breakdown - inf_contained
        du[base + Cleared] += to_cleared - inf_cleared
        du[base + Recovered] += to_rec_from_sub_low + to_rec_from_sub_inf + tx_to_rec - inf_recovered
        du[base + Incipient] += inf_naive + inf_contained + inf_cleared + inf_recovered - to_contained - to_sub_low - to_sub_inf + to_breakdown

        du[base + SubClinLow] += to_sub_low + to_sub_low_from_loss + tx_to_relapse - to_clin_low - to_sub_low_from_gain - to_rec_from_sub_low - to_tx_from_sub_low
        du[base + SubClinInf] += to_sub_inf + to_sub_inf_from_clin_low + to_sub_low_from_gain + to_sub_inf_from_clin_inf - to_clin_inf - to_sub_low_from_loss - to_rec_from_sub_inf - to_tx_from_sub_inf

        du[base + ClinLow] += to_clin_low + to_clin_low_from_loss - to_sub_inf_from_clin_low - to_clin_inf_from_gain - to_tx_from_clin_low - clinlow_death
        du[base + ClinInf] += to_clin_inf + to_clin_inf_from_gain - to_sub_inf_from_clin_inf - to_clin_low_from_loss - to_tx_from_clin_inf - clininf_death

        du[base + Treatment] += to_tx_from_sub_low + to_tx_from_sub_inf + to_tx_from_clin_low + to_tx_from_clin_inf - tx_to_rec - tx_to_relapse - tx_to_death

        du[base + CumInfectionsOther] += inf_naive + inf_cleared + inf_recovered
        du[base + CumInfectionsContained] += inf_contained
        du[base + CumProgressionToActiveTB] += to_sub_low + to_sub_inf
        du[base + CumTreatmentInitiation] += to_tx_from_sub_low + to_tx_from_sub_inf + to_tx_from_clin_low + to_tx_from_clin_inf
        du[base + CumTreatmentCompletion] += tx_to_rec
        du[base + CumRelapseTB] += tx_to_relapse
    end

    return nothing
end

function tb_rhs!(du, u, p::TBParams, t)
    fill!(du, 0.0)
    tb_rhs_epi!(du, u, p, t)
    apply_demography!(du, u, p, t)
    return nothing
end

function simulate_demo(; population::AbstractVector{<:Real} = default_population(),
    contact::AbstractMatrix{<:Real} = default_contact_matrix(),
    tspan::Tuple{Real,Real} = (2025.0, 2030.0),
    saveat::Real = 0.25,
    reltol::Real = 1e-8,
    abstol::Real = 1e-10,
    demography = nothing,
    ageing_enabled::Bool = false,
    kwargs...)
    params = make_parameters(contact; demography = demography, ageing_enabled = ageing_enabled, kwargs...)
    u0 = initial_state(population)
    prob = ODEProblem(tb_rhs!, u0, (Float64(tspan[1]), Float64(tspan[2])), params)
    return solve(prob, Rosenbrock23(autodiff = AutoFiniteDiff()); reltol = reltol, abstol = abstol, saveat = saveat)
end

function simulate_demographic_demo(; population::AbstractVector{<:Real} = default_population(),
    contact::AbstractMatrix{<:Real} = default_contact_matrix(),
    schedule::DemographicSchedule = synthetic_demographic_schedule(),
    tspan::Tuple{Real,Real} = (2025.0, 2030.0),
    saveat::Real = 0.25,
    reltol::Real = 1e-8,
    abstol::Real = 1e-10,
    kwargs...)
    params = make_demographic_parameters(contact, schedule; kwargs...)
    u0 = initial_state(population)
    prob = ODEProblem(tb_rhs!, u0, (Float64(tspan[1]), Float64(tspan[2])), params)
    return solve(prob, Rosenbrock23(autodiff = AutoFiniteDiff()); reltol = reltol, abstol = abstol, saveat = saveat)
end
