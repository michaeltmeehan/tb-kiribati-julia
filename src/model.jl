using LinearAlgebra: I, mul!

const CONTACT = readdlm(".\\data\\contact", Float64)

const OLDEST_AGE = 95
const NAGE = OLDEST_AGE + 1   # Number of age groups: [0,1), [1,2), ..., [94,95), [95,inf)
const NEPI = 10   # Number of epi compartments
const NCUM = 6    # Number of cumulative flow / auxiliary compartments
const NSTATE = NEPI + NCUM

# Compartment encoding / indexing
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

export OLDEST_AGE, NAGE, NEPI, NCUM, NSTATE
export MtbNaive, Contained, Cleared, Recovered, Incipient, SubClinLow, SubClinInf, ClinLow, ClinInf, Treatment
export CumInfectionsOther, CumInfectionsContained, CumProgressionToActiveTB, CumTreatmentInitiation, CumTreatmentCompletion, CumRelapseTB
export TBParams, make_parameters, make_default_parameters, default_contact_matrix, default_population, initial_state
export compute_force_of_infection!, tb_rhs!, apply_demography!

function compute_force_of_infection!(λ::AbstractVector{<:Real}, u::AbstractVector, p::TBParams)
    length(λ) == NAGE || error("λ must have length ", NAGE)
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

function tb_rhs!(du, u, p::TBParams, t)
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

        containment = p.containment_rate[a] * incipient
        clearance = p.clearance_rate * contained
        breakdown = p.breakdown_rate * contained

        progression_to_sublow = (1 - p.prop_infectious) * p.progression_rate[a] * incipient
        progression_to_subinf = p.prop_infectious * p.progression_rate[a] * incipient

        clin_progression_to_clinlow = p.clinical_progression_rate * sublow
        clin_progression_to_clininf = p.clinical_progression_rate * subinf
        clin_regression_to_sublow = p.clinical_regression_rate * clinlow
        clin_regression_to_subinf = p.clinical_regression_rate * clininf
        inf_gain_to_subinf = p.infectiousness_gain_rate * sublow
        inf_gain_to_clininf = p.infectiousness_gain_rate * clinlow
        inf_loss_to_sublow = p.infectiousness_loss_rate * subinf
        inf_loss_to_clinlow = p.infectiousness_loss_rate * clininf

        to_rec_from_sublow = p.self_recovery_rate * sublow
        to_rec_from_subinf = p.self_recovery_rate * subinf

        to_tx_from_sublow = p.rel_detection_subclin * p.detection_rate * sublow
        to_tx_from_subinf = p.rel_detection_subclin * p.detection_rate * subinf
        to_tx_from_clinlow = p.detection_rate * clinlow
        to_tx_from_clininf = p.detection_rate * clininf

        tx_to_rec = p.tx_recovery_rate * tx
        tx_to_relapse = p.tx_relapse_rate * tx
        
        tx_death = p.tx_death_rate * tx
        clinlow_death = p.disease_mortality_clin_lowinf * clinlow
        clininf_death = p.disease_mortality_clin_inf * clininf

        du[base + MtbNaive] = -inf_naive

        du[base + Contained] = containment - clearance - breakdown - inf_contained
        du[base + Cleared] = clearance - inf_cleared
        du[base + Recovered] = to_rec_from_sublow + to_rec_from_subinf + tx_to_rec - inf_recovered
        du[base + Incipient] = inf_naive + inf_contained + inf_cleared + inf_recovered - containment - progression_to_sublow - progression_to_subinf + breakdown

        du[base + SubClinLow] = progression_to_sublow + clin_regression_to_sublow + inf_loss_to_sublow + tx_to_relapse - clin_progression_to_clinlow - inf_gain_to_subinf - to_rec_from_sublow - to_tx_from_sublow
        du[base + SubClinInf] = progression_to_subinf + clin_regression_to_subinf + inf_gain_to_subinf - clin_progression_to_clininf - inf_loss_to_sublow - to_rec_from_subinf - to_tx_from_subinf

        du[base + ClinLow] = clin_progression_to_clinlow + inf_loss_to_clinlow - clin_regression_to_sublow - inf_gain_to_clininf - to_tx_from_clinlow - clinlow_death
        du[base + ClinInf] = clin_progression_to_clininf + inf_gain_to_clininf - clin_regression_to_subinf - inf_loss_to_clinlow - to_tx_from_clininf - clininf_death

        du[base + Treatment] = to_tx_from_sublow + to_tx_from_subinf + to_tx_from_clinlow + to_tx_from_clininf - tx_to_rec - tx_to_relapse - tx_death

        du[base + CumInfectionsOther] = inf_naive + inf_cleared + inf_recovered
        du[base + CumInfectionsContained] = inf_contained
        du[base + CumProgressionToActiveTB] = progression_to_sublow + progression_to_subinf
        du[base + CumTreatmentInitiation] = to_tx_from_sublow + to_tx_from_subinf + to_tx_from_clinlow + to_tx_from_clininf
        du[base + CumTreatmentCompletion] = tx_to_rec
        du[base + CumRelapseTB] = tx_to_relapse
    end

    return nothing
end
