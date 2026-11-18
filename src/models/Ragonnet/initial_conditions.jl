
"""
    initial_state(
        model::RagonnetTBModel,
        params::RagonnetTBParameters,
        demography::AbstractDemography;
        year,
        turnover_rate=nothing,
    )

Construct an initial condition from the analytical endemic
equilibrium of a homogeneous Ragonnet model.

The homogeneous approximation removes:
  - reinfection of previously infected individuals;
  - treatment relapse.

It retains endogenous reactivation, progression, recovery,
detection, treatment and TB mortality.

Age-dependent parameters are population-weighted, and the
homogeneous contact rate preserves the exclusion of
infectious contributions from children aged under 15.

The equilibrium compartment proportions are applied to the
age-specific population. Cumulative counters start at zero.

This is not an exact equilibrium of the age-structured model.
"""
function initial_state(
    model::RagonnetTBModel,
    params::RagonnetTBParameters,
    demography::AbstractDemography;
    year::Real,
    turnover_rate::Union{Nothing,Real}=nothing,
)
    layout = state_layout(model)
    idx = RagonnetIndices

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = get_population(demography, year)

    length(population) == layout.nages ||
        throw(DimensionMismatch(
            "Population length does not match model age classes"
        ))

    all(x -> isfinite(x) && x >= 0, population) ||
        throw(ArgumentError(
            "Population must be finite and nonnegative"
        ))

    total_population = sum(population)

    total_population > 0 ||
        throw(ArgumentError(
            "Initial population must be positive"
        ))

    weights = population ./ total_population

    # --------------------------------------------------------
    # Homogeneous demographic turnover
    # --------------------------------------------------------

    # Approximate the homogeneous background mortality hazard
    # by the population-weighted age-specific mortality rate.
    #
    # Homogeneous births replace background and TB deaths,
    # maintaining a constant total population.

    if isnothing(turnover_rate)
        hasproperty(demography, :data) ||
            throw(ArgumentError(
                "Demography must provide mortality data, " *
                    "or turnover_rate must be specified"
            ))

        data = demography.data

        μ = sum(
            weights[a] * mortality_rate(
                data, a - 1, year
            )
            for a in 1:layout.nages
        )
    else
        μ = Float64(turnover_rate)
    end

    isfinite(μ) && μ > 0 ||
        throw(ArgumentError(
            "Homogeneous turnover rate must be positive"
        ))

    # --------------------------------------------------------
    # Homogeneous epidemiological parameters
    # --------------------------------------------------------

    p = sum(weights .* params.progression_rate)
    c = sum(weights .* params.containment_rate)

    δ = params.clearance_rate
    κ = params.breakdown_rate
    f = params.prop_infectious

    h = params.clinical_progression_rate
    g = params.clinical_regression_rate

    a = params.infectiousness_gain_rate
    b = params.infectiousness_loss_rate

    r = params.self_recovery_rate
    d = params.detection_rate
    α = params.rel_detection_subclin

    tR = params.tx_recovery_rate
    tD = params.tx_death_rate

    mI = params.disease_mortality_clin_lowinf
    mJ = params.disease_mortality_clin_inf

    rates = (
        p, c, δ, κ, h, g, a, b, r, d, α,
        tR, tD, mI, mJ,
    )

    all(x -> isfinite(x) && x >= 0, rates) ||
        throw(ArgumentError(
            "Invalid homogeneous transition rates"
        ))

    0 <= f <= 1 ||
        throw(ArgumentError(
            "prop_infectious must lie in [0, 1]"
        ))

    # --------------------------------------------------------
    # Homogeneous transmission coefficient
    # --------------------------------------------------------

    # In the full Ragonnet model, infectious prevalence
    # is zeroed for source ages below 15.
    #
    # Assuming the same compartment proportions at every age,
    # average the resulting force of infection across recipients.

    size(model.contact) ==
        (layout.nages, layout.nages) ||
        throw(DimensionMismatch(
            "Contact matrix dimension mismatch"
        ))

    contact_factor = 0.0

    for recipient in 1:layout.nages
        for source in 1:layout.nages
            if source - 1 >= 15
                contact_factor +=
                    weights[recipient] *
                        model.contact[recipient, source]
            end
        end
    end

    β = params.beta * contact_factor

    isfinite(β) && β >= 0 ||
        throw(ArgumentError(
            "Invalid homogeneous transmission coefficient"
        ))

    # --------------------------------------------------------
    # Active-disease equilibrium
    # --------------------------------------------------------

    # Solve:
    #
    # M * q = p * [1-f, f, 0, 0]
    #
    # where q gives U, V, I and J relative to E.
    #
    # Treatment relapse is omitted from this equilibrium.

    AU = h + a + r + α*d + μ
    AV = h + b + r + α*d + μ
    AI = g + a + d + mI + μ
    AJ = g + b + d + mJ + μ

    M = [
        AU -b -g 0.0
        -a AV 0.0 -g
        -h 0.0 AI -b
        0.0 -h -a AJ
    ]

    q = M \ (p .* [1 - f, f, 0.0, 0.0])

    all(x -> isfinite(x) && x >= 0, q) ||
        throw(ArgumentError(
            "Invalid active-disease equilibrium"
        ))

    qU, qV, qI, qJ = q

    # Treatment relative to incipient disease.
    qT = d * (α*(qU + qV) + qI + qJ) /
        (μ + tR + tD)

    # Effective infectiousness per incipient individual.
    wU, wV, wI, wJ = params.infectiousness_weights

    W = wU*qU + wV*qV + wI*qI + wJ*qJ

    # Recovery and TB mortality per incipient individual.
    Q = r*(qU + qV) + tR*qT
    D = mI*qI + mJ*qJ + tD*qT

    # --------------------------------------------------------
    # Endemic equilibrium
    # --------------------------------------------------------

    # Containment, clearance and endogenous reactivation.
    A = μ + δ + κ

    # Effective loss rate from incipient infection.
    H = p + μ + c*(μ + δ)/A

    # Disease-free state by default.
    proportions = zeros(Float64, layout.nepi)
    proportions[idx.MtbNaive] = 1.0

    if β > 0 && W > 0 && β*W > H

        R0 = β*W/H

        # Explicit endemic force of infection.
        λ = μ * H * (R0 - 1) / (H - D)

        e = λ / (β*W)

        contained = c*e/A
        cleared = δ*contained/μ
        recovered = Q*e/μ

        sublow = qU*e
        subinf = qV*e
        clinlow = qI*e
        clininf = qJ*e
        treatment = qT*e

        susceptible = 1/R0

        proportions[idx.MtbNaive] = susceptible
        proportions[idx.Contained] = contained
        proportions[idx.Cleared] = cleared
        proportions[idx.Recovered] = recovered
        proportions[idx.Incipient] = e
        proportions[idx.SubClinLow] = sublow
        proportions[idx.SubClinInf] = subinf
        proportions[idx.ClinLow] = clinlow
        proportions[idx.ClinInf] = clininf
        proportions[idx.Treatment] = treatment
    end

    # --------------------------------------------------------
    # Validate equilibrium proportions
    # --------------------------------------------------------

    all(isfinite, proportions) ||
        throw(ArgumentError(
            "Nonfinite equilibrium proportions"
        ))

    all(x -> x >= -1e-12, proportions) ||
        throw(ArgumentError(
            "Negative equilibrium compartment proportion"
        ))

    isapprox(
        sum(proportions), 1.0;
        atol=1e-10,
        rtol=1e-10,
    ) || throw(ArgumentError(
        "Homogeneous equilibrium does not conserve population"
    ))

    # --------------------------------------------------------
    # Construct age-specific initial state
    # --------------------------------------------------------

    u0 = zeros(
        Float64,
        nstates(layout) * layout.nages,
    )

    @inbounds for age in 1:layout.nages
        N = population[age]
        base = (age - 1)*nstates(layout)

        for compartment in 1:layout.nepi
            u0[base+compartment] =
                N * proportions[compartment]
        end
    end

    return u0
end
