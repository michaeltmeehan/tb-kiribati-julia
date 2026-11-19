
using LinearAlgebra: mul!

function compute_force_of_infection!(
    u::AbstractVector,
    ctx::SimulationContext{<:DoddExtendedTBModel},
)
    model = ctx.model
    p = ctx.params
    layout = model.layout
    work = ctx.workspace

    nstate = nstates(layout)
    q = p.relative_infectiousness

    @inbounds for a in 1:layout.nages
        base = (a - 1) * nstate

        population = 0.0
        for c in 1:layout.nepi
            population += u[base+c]
        end

        infectious =
            q[1] * (
                    u[base+DE.I0SP] +
                        u[base+DE.I1SP]
                ) +
                q[2] * (
                    u[base+DE.I0SN] +
                        u[base+DE.I1SN]
                ) +
                q[3] * (
                    u[base+DE.I0EP] +
                        u[base+DE.I1EP]
                )

        work.infectious_prevalence[a] =
            population > 0 ?
            model.infectiousness[a] *
                infectious / population :
            0.0
    end

    mul!(
        work.foi,
        model.contact,
        work.infectious_prevalence,
    )

    @inbounds for a in 1:layout.nages
        work.foi[a] *= p.beta
    end

    return nothing
end


function tb_rhs!(
    du,
    u,
    ctx::SimulationContext{<:DoddExtendedTBModel},
    t,
)
    model = ctx.model
    p = ctx.params

    layout = model.layout

    compute_force_of_infection!(u, ctx)

    nstate = nstates(layout)

    probability = detection_probability(
        p.detection, t
    )

    detection_odds = probability / (1 - probability)

    τ = p.treatment_rate
    θ = p.treatment_cfr

    @inbounds for a in 1:layout.nages
        base = (a - 1) * nstate
        λ = ctx.workspace.foi[a]

        # Epidemiological states.
        susceptible = u[base+DE.S]
        early_latent_0 = u[base+DE.E0]
        early_latent_1 = u[base+DE.E1]
        late_latent_0 = u[base+DE.L0]
        late_latent_1 = u[base+DE.L1]
        treatment = u[base+DE.T]
        recovered = u[base+DE.R]

        # Infection and reinfection.
        primary_infections = λ * susceptible

        reinfections_latent_0 =
            p.psi * λ * late_latent_0

        reinfections_latent_1 =
            p.psi * λ * late_latent_1

        reinfections_recovered =
            p.psi * λ * recovered

        # Latent TB transitions.
        early_to_late_0 =
            p.kappa * early_latent_0

        early_to_late_1 =
            p.kappa * early_latent_1

        fast_progression_0 =
            p.epsilon[a] * early_latent_0

        fast_progression_1 =
            p.epsilon[a] * early_latent_1

        slow_progression_0 =
            p.nu[a] * late_latent_0

        slow_progression_1 =
            p.nu[a] * late_latent_1

        relapse = p.omega[a] * recovered

        onset_0 =
            fast_progression_0 + slow_progression_0

        onset_1 =
            fast_progression_1 +
                slow_progression_1 +
                relapse

        # Active disease transitions.
        self_cure_0 = 0.0
        self_cure_1 = 0.0

        untreated_deaths = 0.0
        treatment_initiation = 0.0

        base_detection =
            p.detection_multiplier[a] *
                detection_odds

        for k in 1:3
            i0 = DE.I0SP + k - 1
            i1 = DE.I1SP + k - 1

            active_0 = u[base+i0]
            active_1 = u[base+i1]

            δ = base_detection *
                p.relative_detection[k]

            γ = p.self_cure[a]
            μ = p.tb_mortality[a]

            detection_0 = δ * active_0
            detection_1 = δ * active_1

            cure_0 = γ * active_0
            cure_1 = γ * active_1

            deaths = μ * (active_0 + active_1)

            self_cure_0 += cure_0
            self_cure_1 += cure_1

            untreated_deaths += deaths

            treatment_initiation +=
                detection_0 + detection_1

            presentation = p.presentation[a, k]

            du[base+i0] =
                presentation * onset_0 -
                    detection_0 - cure_0 -
                    μ * active_0

            du[base+i1] =
                presentation * onset_1 -
                    detection_1 - cure_1 -
                    μ * active_1

            # New notifications.
            du[base+DE.CumNewSP+k-1] =
                detection_0

            # Recurrent notifications.
            du[base+DE.CumRecurrentSP+k-1] =
                detection_1
        end

        # Treatment outcomes.
        treatment_completion =
            (1 - θ) * τ * treatment

        treatment_deaths =
            θ * τ * treatment

        # Epidemiological equations.
        du[base+DE.S] =
            -primary_infections

        du[base+DE.E0] =
            primary_infections +
                reinfections_latent_0 -
                early_to_late_0 -
                fast_progression_0

        du[base+DE.E1] =
            reinfections_latent_1 +
                reinfections_recovered -
                early_to_late_1 -
                fast_progression_1

        du[base+DE.L0] =
            early_to_late_0 +
                self_cure_0 -
                reinfections_latent_0 -
                slow_progression_0

        du[base+DE.L1] =
            early_to_late_1 +
                self_cure_1 -
                reinfections_latent_1 -
                slow_progression_1

        du[base+DE.T] =
            treatment_initiation -
                treatment_completion -
                treatment_deaths

        du[base+DE.R] =
            treatment_completion -
                relapse -
                reinfections_recovered

        # Cumulative event equations.
        du[base+DE.CumPrimaryInfections] =
            primary_infections

        du[base+DE.CumReinfections] =
            reinfections_latent_0 +
                reinfections_latent_1 +
                reinfections_recovered

        du[base+DE.CumFastProgression] =
            fast_progression_0 +
                fast_progression_1

        du[base+DE.CumSlowProgression] =
            slow_progression_0 +
                slow_progression_1

        du[base+DE.CumRelapse] = relapse

        du[base+DE.CumTreatmentInitiation] =
            treatment_initiation

        du[base+DE.CumTreatmentCompletion] =
            treatment_completion

        du[base+DE.CumUntreatedTBDeaths] =
            untreated_deaths

        du[base+DE.CumTreatmentTBDeaths] =
            treatment_deaths

        du[base+DE.CumSelfCure] =
            self_cure_0 + self_cure_1
    end

    return nothing
end
