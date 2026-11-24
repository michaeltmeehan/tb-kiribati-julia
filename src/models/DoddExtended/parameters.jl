
using Distributions: Beta, LogNormal, Normal

struct DoddExtendedDetection{T<:Real}
    K::T
    trend::T
    reference_year::T
    maximum::T
end

function DoddExtendedDetection(;
    K::Real,
    trend::Real,
    reference_year::Real=2019.0,
    maximum::Real=0.9,
)
    0 < K < 1 ||
        throw(ArgumentError("K must lie in (0, 1)"))

    0 < maximum < 1 ||
        throw(ArgumentError("maximum must lie in (0, 1)"))

    k, tr, yr, mx = promote(
        float(K), float(trend),
        float(reference_year), float(maximum),
    )

    return DoddExtendedDetection(k, tr, yr, mx)
end

@inline function detection_probability(
    d::DoddExtendedDetection, t
)
    z = log(d.K / (1 - d.K)) +
        d.trend * (t - d.reference_year)

    logistic = if z >= 0
        inv(1 + exp(-z))
    else
        ez = exp(z)
        ez / (1 + ez)
    end

    return d.maximum * logistic
end

struct DoddExtendedTBParameters{T<:Real}
    Lambda0::T
    beta::T
    psi::T
    kappa::Vector{T}

    epsilon::Vector{T}
    nu::Vector{T}
    omega::Vector{T}

    self_cure::Matrix{T}
    tb_mortality::Matrix{T}
    detection_multiplier::Vector{T}

    treatment_rate::T
    treatment_cfr::T
    detection::DoddExtendedDetection{T}

    presentation_fast::Matrix{T}
    presentation_slow::Matrix{T}
    presentation_relapse::Matrix{T}
    relative_infectiousness::NTuple{3,T}
    relative_detection::NTuple{3,T}
end


function priors(model::DoddExtendedTBModel)
    p = (
        # Transmission and reinfection
        beta=LogNormal(log(10), 0.5),
        Lambda0=LogNormal(log(0.005), 0.7),
        psi=Beta(20.7, 77.9),
        smear_negative_infectiousness=LogNormal(log(0.35), 0.4),

        # Progression and recurrence
        early_duration=LogNormal(log(0.5), 0.35),
        fast_risk_0=Beta(9, 11),
        fast_risk_1=Beta(3.6, 16.4),
        fast_risk_2_4=Beta(1.1, 18.9),
        fast_risk_5_9=Beta(0.4, 19.6),
        fast_risk_10_14=Beta(2, 18),
        fast_risk_15_plus=Beta(1, 19),
        nu=LogNormal(log(0.003), 1.0),
        omega=LogNormal(log(0.005), 0.5),

        # Detection
        detection_logitK=Normal(-0.4, 0.8),
        detection_trend=Normal(0.0, 0.03),
        r_SP=LogNormal(log(2.5), 0.5),
        smear_negative_detection=LogNormal(0.0, 0.5),
        extrapulmonary_detection=LogNormal(0.0, 0.5),
        detection_or_0_4=LogNormal(-0.100, 0.626),
        detection_or_5_14=LogNormal(-0.567, 0.458),

        # Adult pulmonary natural history
        mortality_sp=LogNormal(log(0.389), 0.12),
        self_cure_sp=LogNormal(log(0.231), 0.18),
        mortality_sn=LogNormal(log(0.025), 0.25),
        self_cure_sn=LogNormal(log(0.130), 0.30),

        # Childhood mortality relative to adult hazards
        child_mortality_0_4=LogNormal(log(1.0), 0.8),
        child_mortality_5_14=LogNormal(log(0.3), 0.8),

        # Extrapulmonary natural history
        duration_ep=LogNormal(log(3.0), 0.40),
        untreated_cfr_ep=Beta(4, 16),

        # Treatment
        treatment_cfr=Beta(2.71, 87.55),
    )

    return model.reinfection ?
           p : Base.structdiff(p, (psi=nothing,))
end


extended_fixed_parameters() = (
    smear_negative_infectiousness=0.35,
    smear_negative_detection=1.0,
    extrapulmonary_detection=1.0,
)

function fast_progression_hazards(
    risk::AbstractVector{T},
    duration::Real,
) where {T<:Real}
    duration > 0 ||
        throw(ArgumentError("duration must be positive"))

    all(p -> 0 <= p < 1, risk) ||
        throw(ArgumentError("risks must be in [0, 1)"))

    epsilon = risk ./ duration
    kappa = (one(T) .- risk) ./ duration

    return epsilon, kappa
end

function expand_fast_progression_risks(r, nages)
    length(r) == 6 ||
        throw(ArgumentError("Expected six age-band risks"))

    return [
        age == 0 ? r[1] :
        age == 1 ? r[2] :
        age < 5 ? r[3] :
        age < 10 ? r[4] :
        age < 15 ? r[5] :
        r[6]
        for age in 0:(nages-1)
    ]
end

function adjust_ep_presentation(
    P::AbstractMatrix,
    odds_ratio::Real;
    max_age::Int=4,
)
    isfinite(odds_ratio) && odds_ratio > 0 ||
        throw(ArgumentError(
            "EP odds ratio must be finite and positive"
        ))

    size(P, 2) == 3 ||
        throw(DimensionMismatch(
            "Expected SP, SN and EP columns"
        ))

    Q = copy(P)

    for a in 1:min(size(Q, 1), max_age+1)
        Q[a, 3] *= odds_ratio
        Q[a, :] ./= sum(@view Q[a, :])
    end

    return Q
end

function make_parameters(
    model::DoddExtendedTBModel,
    draw::NamedTuple;
    fast_ep_odds_under5::Real=1.0,
)
    nages = state_layout(model).nages

    # Promote every sampled quantity to a common numerical
    # type, including ForwardDiff.Dual when applicable.
    scalar_values = (
        v for v in Base.values(draw) if v isa Real
    )

    T = promote_type(
        Float64,
        (typeof(v) for v in scalar_values)...,
    )

    x(name) = T(getproperty(draw, name))

    Λ0 = x(:Lambda0)
    β = x(:beta)
    ψ = model.reinfection ? x(:psi) : zero(T)

    risk_bands = T[
        x(:fast_risk_0),
        x(:fast_risk_1),
        x(:fast_risk_2_4),
        x(:fast_risk_5_9),
        x(:fast_risk_10_14),
        x(:fast_risk_15_plus),
    ]

    risk = expand_fast_progression_risks(
        risk_bands, nages
    )

    ε, κ = fast_progression_hazards(
        risk, x(:early_duration)
    )

    ν = fill(x(:nu), nages)
    ω = fill(x(:omega), nages)


    # Adult pulmonary natural-history hazards.
    adult_gamma = T[
        x(:self_cure_sp),
        x(:self_cure_sn),
        (one(T)-x(:untreated_cfr_ep))/x(:duration_ep),
    ]

    adult_mu = T[
        x(:mortality_sp),
        x(:mortality_sn),
        x(:untreated_cfr_ep)/x(:duration_ep),
    ]

    # Presentation-specific natural-history hazards.
    #
    # Adult hazards provide the reference values.
    # Mortality is modified by age, while self-cure
    # retains the adult reference hazard.

    γ = Matrix{T}(undef, nages, 3)
    μ = Matrix{T}(undef, nages, 3)

    m_0_4 = x(:child_mortality_0_4)
    m_5_14 = x(:child_mortality_5_14)

    all(v -> isfinite(v) && v > 0, (m_0_4, m_5_14)) ||
        throw(ArgumentError(
            "Childhood mortality multipliers must be positive"
        ))

    for a in 1:nages
        age = a - 1

        mortality_multiplier =
            age < 5 ? m_0_4 :
            age < 15 ? m_5_14 :
            one(T)

        for k in 1:3
            γ[a, k] = adult_gamma[k]
            μ[a, k] = mortality_multiplier * adult_mu[k]
        end
    end

    relative_detection_age = T[
        age < 5 ? x(:detection_or_0_4) :
        age < 15 ? x(:detection_or_5_14) :
        one(T)
        for age in 0:(nages-1)
    ]

    z = x(:detection_logitK)

    K = if z >= 0
        inv(one(T) + exp(-z))
    else
        ez = exp(z)
        ez / (one(T) + ez)
    end

    detection = DoddExtendedDetection(
        K=K,
        trend=x(:detection_trend),
        reference_year=T(2019),
        maximum=T(0.9),
    )

    q_negative = x(:smear_negative_infectiousness)

    r_SP = x(:r_SP)

    d_negative = x(:smear_negative_detection)
    d_extra = x(:extrapulmonary_detection)

    all(v -> isfinite(v) && v >= 0,
        (Λ0, β, ψ, x(:treatment_cfr))) ||
        throw(ArgumentError("Invalid nonnegative parameter"))

    all(v -> isfinite(v) && v >= 0, κ) ||
        throw(ArgumentError(
            "Invalid stabilisation hazards"
        ))

    all(v -> isfinite(v) && v > 0,
        (adult_gamma..., adult_mu...)) ||
        throw(ArgumentError(
            "Natural-history hazards must be finite and positive"
        ))

    all(v -> isfinite(v) && v > 0,
        (q_negative, r_SP, d_negative, d_extra)) ||
        throw(ArgumentError(
            "Invalid presentation-specific multipliers"
        ))

    0 <= x(:untreated_cfr_ep) <= 1 ||
        throw(ArgumentError("Invalid EP untreated CFR"))

    0 <= x(:treatment_cfr) <= 1 ||
        throw(ArgumentError("Invalid treatment CFR"))

    all(v -> isfinite(v) && v >= 0, ε) ||
        throw(ArgumentError("Invalid fast progression"))

    all(v -> isfinite(v) && v >= 0, ν) ||
        throw(ArgumentError("Invalid slow progression"))

    all(v -> isfinite(v) && v >= 0, ω) ||
        throw(ArgumentError("Invalid relapse rate"))

    all(v -> isfinite(v) && v >= 0, relative_detection_age) ||
        throw(ArgumentError("Invalid detection multipliers"))

    # Detection hazards are not determined by the
    # presentation-specific natural-history durations.
    #
    # Preserve the original reference scale:
    # 0.5 years for children, 3 years for adults.
    detection_reference_duration = T[
        age < 15 ? T(0.5) : T(3.0)
        for age in 0:(nages-1)
    ]

    detection_multiplier =
        relative_detection_age ./ detection_reference_duration

    # Baseline age-specific presentation probabilities.
    baseline = T.(model.presentation)

    # Optional sensitivity modifier for EP disease
    # following fast progression at ages 0–4.
    presentation_fast = adjust_ep_presentation(
        baseline,
        T(fast_ep_odds_under5);
        max_age=4,
    )

    presentation_slow = copy(baseline)
    presentation_relapse = copy(baseline)

    # Validate presentation distributions.
    for P in (
        presentation_fast,
        presentation_slow,
        presentation_relapse,
    )
        size(P) == (nages, 3) ||
            throw(DimensionMismatch(
                "Invalid presentation matrix dimensions"
            ))

        all(v -> isfinite(v) && 0 <= v <= 1, P) ||
            throw(ArgumentError(
                "Invalid presentation probabilities"
            ))

        all(isapprox.(vec(sum(P; dims=2)), one(T);
            atol=1e-10)) ||
            throw(ArgumentError(
                "Presentation probabilities must sum to one"
            ))
    end

    return DoddExtendedTBParameters(
        Λ0, β, ψ, κ,
        ε, ν, ω,
        γ, μ, detection_multiplier,
        T(2), x(:treatment_cfr), detection,
        presentation_fast,
        presentation_slow,
        presentation_relapse,
        (one(T), q_negative, zero(T)),
        (r_SP, d_negative, d_extra),
    )
end

"""
Probability of diagnosis before self-cure or TB death,
conditional on active disease presentation and age.

Assumes constant competing hazards and no ageing
during the disease episode.
"""
function presentation_ascertainment(
    p::DoddExtendedTBParameters,
    year::Real,
)
    nages = length(p.nu)
    T = typeof(p.beta)
    A = Matrix{T}(undef, nages, 3)

    probability = detection_probability(p.detection, year)
    odds = probability / (one(T) - probability)

    for a in 1:nages, k in 1:3
        δ = odds *
            p.detection_multiplier[a] *
            p.relative_detection[k]

        γ = p.self_cure[a, k]
        μ = p.tb_mortality[a, k]

        total = δ + γ + μ

        total > 0 ||
            throw(ArgumentError(
                "Invalid competing hazards"
            ))

        A[a, k] = δ / total
    end

    return A
end


"""
Expected notification proportions given the
underlying disease-onset presentation distribution.
"""
function expected_notification_presentation(
    P::AbstractMatrix,
    ascertainment::AbstractMatrix,
)
    size(P) == size(ascertainment) ||
        throw(DimensionMismatch(
            "Presentation and ascertainment dimensions differ"
        ))

    size(P, 2) == 3 ||
        throw(DimensionMismatch(
            "Expected three disease presentations"
        ))

    all(isfinite, P) && all(x -> x >= 0, P) ||
        throw(ArgumentError("Invalid presentation probabilities"))

    all(isfinite, ascertainment) &&
        all(x -> 0 <= x <= 1, ascertainment) ||
        throw(ArgumentError("Invalid ascertainment probabilities"))

    Q = P .* ascertainment

    for a in axes(Q, 1)
        total = sum(@view Q[a, :])

        total > 0 ||
            throw(ArgumentError(
                "Zero expected notifications at age index $a"
            ))

        Q[a, :] ./= total
    end

    return Q
end

"""
Infer disease-onset presentation proportions from
notification proportions and ascertainment probabilities.

Uses the stationary competing-risk approximation.
This is a diagnostic transformation, not a calibration.
"""
function inferred_onset_presentation(
    Q::AbstractMatrix,
    ascertainment::AbstractMatrix,
)
    size(Q) == size(ascertainment) ||
        throw(DimensionMismatch(
            "Notification and ascertainment dimensions differ"
        ))

    size(Q, 2) == 3 ||
        throw(DimensionMismatch(
            "Expected SP, SN and EP columns"
        ))

    all(x -> isfinite(x) && x >= 0, Q) ||
        throw(ArgumentError(
            "Invalid notification probabilities"
        ))

    all(x -> isfinite(x) && 0 < x <= 1, ascertainment) ||
        throw(ArgumentError(
            "Ascertainment must lie in (0, 1]"
        ))

    all(isapprox.(vec(sum(Q; dims=2)), 1.0; atol=1e-8)) ||
        throw(ArgumentError(
            "Notification probabilities must sum to one"
        ))

    P = Q ./ ascertainment

    for a in axes(P, 1)
        total = sum(@view P[a, :])

        total > 0 ||
            throw(ArgumentError(
                "Zero onset probability at age index $a"
            ))

        P[a, :] ./= total
    end

    return P
end
