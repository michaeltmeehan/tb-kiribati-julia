using LinearAlgebra: mul!

"""
    compute_force_of_infection!(u, ctx)

Calculate the age-specific force of infection:

    λₐ = β ∑ᵦ Cₐᵦ qᵦ Iᵦ / Nᵦ

where:
- Cₐᵦ is the contact rate between ages a and b.
- qᵦ is the relative infectiousness of age group b.
- Iᵦ is the active TB population in age group b.
- Nᵦ is the total living population in age group b.

The results are written in-place to ctx.workspace.foi.
"""
function compute_force_of_infection!(
    u::AbstractVector,
    ctx::SimulationContext{<:DoddTBModel},
)
    model = ctx.model
    p = ctx.params
    layout = model.layout
    work = ctx.workspace

    nages = layout.nages
    nstate = nstates(layout)

    # --------------------------------------------------------
    # 1. Calculate infectious prevalence by age
    # --------------------------------------------------------

    @inbounds for a in 1:nages
        base = (a - 1) * nstate

        population = 0.0

        for c in 1:layout.nepi
            population += u[base + c]
        end

        infectious = u[base + I]

        work.infectious_prevalence[a] =
            population > 0 ?
            model.infectiousness[a] * infectious / population :
            0.0
    end

    # --------------------------------------------------------
    # 2. Calculate age-specific force of infection
    # --------------------------------------------------------

    mul!(
        work.foi,
        model.contact,
        work.infectious_prevalence,
    )

    @inbounds for a in 1:nages
        work.foi[a] *= p.beta
    end

    return nothing
end

function tb_rhs!(
    du,
    u,
    ctx::SimulationContext{<:DoddTBModel},
    t,
)
    model = ctx.model
    p = ctx.params
    layout = model.layout

    compute_force_of_infection!(u, ctx)

    nstate = nstates(layout)
    τ = p.treatment_rate
    θ = p.treatment_cfr

    # Common time-dependent detection odds.
    probability = detection_probability(p.detection, t)
    detection_odds = probability / (1 - probability)

    @inbounds for a in 1:layout.nages
        base = (a - 1) * nstate

        # Epidemiological states
        susceptible = u[base + S]
        early_latent = u[base + E]
        late_latent = u[base + L]
        active = u[base + I]
        treatment = u[base + T]
        recovered = u[base + R]

        λ = ctx.workspace.foi[a]

        # Infection and reinfection
        primary_infections = λ * susceptible
        reinfections_latent = p.psi * λ * late_latent
        reinfections_recovered = p.psi * λ * recovered

        # Latent TB transitions
        early_to_late = p.kappa * early_latent
        fast_progression = p.epsilon[a] * early_latent
        slow_progression = p.nu[a] * late_latent

        # Active TB transitions
        relapse = p.omega[a] * recovered
        self_cure = p.self_cure[a] * active
        untreated_deaths = p.tb_mortality[a] * active

        treatment_initiation =
            p.detection_multiplier[a] * detection_odds * active

        # Treatment outcomes
        treatment_completion = (1 - θ) * τ * treatment
        treatment_deaths = θ * τ * treatment

        # --------------------------------------------------
        # Epidemiological equations
        # --------------------------------------------------

        du[base + S] =
            -primary_infections

        du[base + E] =
            primary_infections +
            reinfections_latent +
            reinfections_recovered -
            early_to_late -
            fast_progression

        du[base + L] =
            early_to_late +
            self_cure -
            reinfections_latent -
            slow_progression

        du[base + I] =
            fast_progression +
            slow_progression +
            relapse -
            self_cure -
            untreated_deaths -
            treatment_initiation

        du[base + T] =
            treatment_initiation -
            treatment_completion -
            treatment_deaths

        du[base + R] =
            treatment_completion -
            relapse -
            reinfections_recovered

        # --------------------------------------------------
        # Cumulative event equations
        # --------------------------------------------------

        du[base + CumPrimaryInfections] =
            primary_infections

        du[base + CumReinfections] =
            reinfections_latent + reinfections_recovered

        du[base + CumFastProgression] =
            fast_progression

        du[base + CumSlowProgression] =
            slow_progression

        du[base + CumRelapse] =
            relapse

        du[base + CumTreatmentInitiation] =
            treatment_initiation

        du[base + CumTreatmentCompletion] =
            treatment_completion

        du[base + CumUntreatedTBDeaths] =
            untreated_deaths

        du[base + CumTreatmentTBDeaths] =
            treatment_deaths

        du[base + CumSelfCure] =
            self_cure
    end

    return nothing
end