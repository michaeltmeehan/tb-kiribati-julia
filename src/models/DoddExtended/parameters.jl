
using Distributions: Beta, Gamma, LogNormal, Normal

struct DoddExtendedDetection{T<:Real}
    K::T
    trend::T
    reference_year::T
    maximum::T
end

function DoddExtendedDetection(;
    K::Real,
    trend::Real,
    reference_year::Real=1970.0,
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
    kappa::T

    epsilon::Vector{T}
    nu::Vector{T}
    omega::Vector{T}

    self_cure::Matrix{T}
    tb_mortality::Matrix{T}
    detection_multiplier::Vector{T}

    treatment_rate::T
    treatment_cfr::T
    detection::DoddExtendedDetection{T}

    presentation::Matrix{T}
    relative_infectiousness::NTuple{3,T}
    relative_detection::NTuple{3,T}
end

function priors(model::DoddExtendedTBModel)
    base = (
        beta=LogNormal(log(10), 0.75),
        psi=Beta(20.7, 77.9),
        Lambda0=LogNormal(log(3e-2), 0.75),
        kappa=LogNormal(0.62, 0.068),
        epsilon=LogNormal(-2.837, 0.32),
        epsilon_child=Beta(5.153, 21.967),
        nu=LogNormal(-6.89, 0.58),
        omega=LogNormal(-3.95, 0.27),
        detection_logitK=Normal(0.0, 0.3),
        detection_trend=Gamma(1., 0.02),
        detection_or_0_4=LogNormal(-0.100, 0.626),
        detection_or_5_14=LogNormal(-0.567, 0.458),
        duration_sp=LogNormal(log(1.6), 0.20),
        duration_sn=LogNormal(log(5.35), 0.30),
        duration_ep=LogNormal(log(3.0), 0.40),
        untreated_cfr_sp=Beta(14, 6),
        untreated_cfr_sn=Beta(4, 16),
        untreated_cfr_ep=Beta(4, 16),
        treatment_cfr=Beta(2.71, 87.55),
    )

    if !model.reinfection
        keys_without_psi = Tuple(
            key for key in keys(base) if key != :psi
        )
        base = NamedTuple{keys_without_psi}(
            Tuple(getproperty(base, k) for k in keys_without_psi)
        )
    end

    return merge(base, (
        smear_negative_infectiousness=
        LogNormal(log(0.35), 0.4),
        smear_negative_detection=
        LogNormal(0.0, 0.5),
        extrapulmonary_detection=
        LogNormal(0.0, 0.5),
    ))
end

extended_fixed_parameters() = (
    smear_negative_infectiousness=0.35,
    smear_negative_detection=1.0,
    extrapulmonary_detection=1.0,
)

function make_parameters(
    model::DoddExtendedTBModel,
    draw::NamedTuple,
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
    κ = x(:kappa)

    ε = T[
        age < 5 ? x(:epsilon_child) : x(:epsilon)
        for age in 0:(nages-1)
    ]

    ν = fill(x(:nu), nages)
    ω = fill(x(:omega), nages)

    adult_duration = T[
        x(:duration_sp),
        x(:duration_sn),
        x(:duration_ep),
    ]

    # Preserve the existing 0.5-year childhood assumption.
    # Presentation-specific durations apply from age 15 onward.
    duration = Matrix{T}(undef, nages, 3)

    for a in 1:nages, k in 1:3
        duration[a, k] =
            a <= 15 ? T(0.5) : adult_duration[k]
    end

    fatality = T[
        x(:untreated_cfr_sp),
        x(:untreated_cfr_sn),
        x(:untreated_cfr_ep),
    ]

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
        reference_year=T(1970),
        maximum=T(0.9),
    )

    q_negative = x(:smear_negative_infectiousness)
    d_negative = x(:smear_negative_detection)
    d_extra = x(:extrapulmonary_detection)

    all(v -> isfinite(v) && v >= 0,
        (Λ0, β, ψ, κ, x(:treatment_cfr))) ||
        throw(ArgumentError("Invalid nonnegative parameter"))

    all(c -> isfinite(c) && 0 <= c <= 1, fatality) ||
        throw(ArgumentError(
            "Untreated CFRs must be finite and between 0 and 1"
        ))

    0 <= x(:treatment_cfr) <= 1 ||
        throw(ArgumentError("Invalid treatment CFR"))

    all(v -> isfinite(v) && v > 0,
        (adult_duration...,
            q_negative, d_negative, d_extra)) ||
        throw(ArgumentError(
            "Invalid untreated duration or presentation parameter"
        ))

    all(v -> isfinite(v) && v >= 0, ε) ||
        throw(ArgumentError("Invalid fast progression"))

    all(v -> isfinite(v) && v >= 0, ν) ||
        throw(ArgumentError("Invalid slow progression"))

    all(v -> isfinite(v) && v >= 0, ω) ||
        throw(ArgumentError("Invalid relapse rate"))

    all(v -> isfinite(v) && v >= 0, relative_detection_age) ||
        throw(ArgumentError("Invalid detection multipliers"))

    γ = (one(T) .- reshape(fatality, 1, 3)) ./ duration
    μ = reshape(fatality, 1, 3) ./ duration

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

    return DoddExtendedTBParameters(
        Λ0, β, ψ, κ,
        ε, ν, ω,
        γ, μ, detection_multiplier,
        T(2), x(:treatment_cfr), detection,
        T.(model.presentation),
        (one(T), q_negative, zero(T)),
        (one(T), d_negative, d_extra),
    )
end
