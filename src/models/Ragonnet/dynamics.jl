# Ragonnet differential equations.
# Migrated from legacy_dynamics.jl.
# State indices are model-specific.
# Dimensions are obtained from StateLayout.
# Equations and parameter values are unchanged.

function ragonnet_force_of_infection!(
    λ::AbstractVector{<:Real},
    u::AbstractVector,
    p::RagonnetTBParameters,
    layout::StateLayout,
    workspace::TBWorkspace,
)
    length(λ) == layout.nages || error("λ must have length ", layout.nages)
    q = workspace.infectious_prevalence
    @inbounds for a in 1:layout.nages
        age = a - 1
        base = (a - 1) * nstates(layout)
        if age < 15
            q[a] = 0.0
        else
            # rel_infectiousness_by_age = age < 15 ? 0.2 : 1.0
            total = 0.0
            for c in 1:layout.nepi
                total += u[base+c]
            end
            if total > 0
                weighted = p.infectiousness_weights[1] * u[base+RagonnetIndices.SubClinLow] +
                    p.infectiousness_weights[2] * u[base+RagonnetIndices.SubClinInf] +
                    p.infectiousness_weights[3] * u[base+RagonnetIndices.ClinLow] +
                    p.infectiousness_weights[4] * u[base+RagonnetIndices.ClinInf]
                q[a] = weighted / total
                # q[a] = rel_infectiousness_by_age * weighted / total
            else
                q[a] = 0.0
            end
        end
    end
    mul!(λ, p.contact, q)
    @inbounds for a in 1:layout.nages
        λ[a] *= p.beta
    end
    return λ
end

function ragonnet_rhs!(
    du,
    u,
    p::RagonnetTBParameters,
    t,
    layout::StateLayout,
    workspace::TBWorkspace,
)
    λ = workspace.foi
    ragonnet_force_of_infection!(
        λ, u, p, layout, workspace
    )

    @inbounds for a in 1:layout.nages
        base = (a - 1) * nstates(layout)
        λa = λ[a]

        naive = u[base+RagonnetIndices.MtbNaive]
        contained = u[base+RagonnetIndices.Contained]
        cleared = u[base+RagonnetIndices.Cleared]
        recovered = u[base+RagonnetIndices.Recovered]
        incipient = u[base+RagonnetIndices.Incipient]
        sublow = u[base+RagonnetIndices.SubClinLow]
        subinf = u[base+RagonnetIndices.SubClinInf]
        clinlow = u[base+RagonnetIndices.ClinLow]
        clininf = u[base+RagonnetIndices.ClinInf]
        tx = u[base+RagonnetIndices.Treatment]

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

        du[base+RagonnetIndices.MtbNaive] = -inf_naive

        du[base+RagonnetIndices.Contained] = containment - clearance - breakdown - inf_contained
        du[base+RagonnetIndices.Cleared] = clearance - inf_cleared
        du[base+RagonnetIndices.Recovered] = to_rec_from_sublow + to_rec_from_subinf + tx_to_rec - inf_recovered
        du[base+RagonnetIndices.Incipient] = inf_naive + inf_contained + inf_cleared + inf_recovered - containment - progression_to_sublow - progression_to_subinf + breakdown

        du[base+RagonnetIndices.SubClinLow] = progression_to_sublow + clin_regression_to_sublow + inf_loss_to_sublow + tx_to_relapse - clin_progression_to_clinlow - inf_gain_to_subinf - to_rec_from_sublow - to_tx_from_sublow
        du[base+RagonnetIndices.SubClinInf] = progression_to_subinf + clin_regression_to_subinf + inf_gain_to_subinf - clin_progression_to_clininf - inf_loss_to_sublow - to_rec_from_subinf - to_tx_from_subinf

        du[base+RagonnetIndices.ClinLow] = clin_progression_to_clinlow + inf_loss_to_clinlow - clin_regression_to_sublow - inf_gain_to_clininf - to_tx_from_clinlow - clinlow_death
        du[base+RagonnetIndices.ClinInf] = clin_progression_to_clininf + inf_gain_to_clininf - clin_regression_to_subinf - inf_loss_to_clinlow - to_tx_from_clininf - clininf_death

        du[base+RagonnetIndices.Treatment] = to_tx_from_sublow + to_tx_from_subinf + to_tx_from_clinlow + to_tx_from_clininf - tx_to_rec - tx_to_relapse - tx_death

        du[base+RagonnetIndices.CumInfectionsOther] = inf_naive + inf_cleared + inf_recovered
        du[base+RagonnetIndices.CumInfectionsContained] = inf_contained
        du[base+RagonnetIndices.CumProgressionToActiveTB] = progression_to_sublow + progression_to_subinf
        du[base+RagonnetIndices.CumTreatmentInitiation] = to_tx_from_sublow + to_tx_from_subinf + to_tx_from_clinlow + to_tx_from_clininf
        du[base+RagonnetIndices.CumTreatmentCompletion] = tx_to_rec
        du[base+RagonnetIndices.CumRelapseTB] = tx_to_relapse
        du[base+RagonnetIndices.CumDeathsTB] = clinlow_death + clininf_death + tx_death
    end

    return nothing
end
