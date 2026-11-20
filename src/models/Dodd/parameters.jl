# ------------------------------------------------------------
# Detection trajectory
# ------------------------------------------------------------

struct DoddDetection{T<:Real}
    K::T
    trend::T
    reference_year::T
    maximum::T
end

function DoddDetection(;
    K::Real,
    trend::Real,
    reference_year::Real=1970.0,
    maximum::Real=0.9,
)
    0 < K < 1 ||
        throw(ArgumentError("K must lie in (0, 1)"))

    0 < maximum < 1 ||
        throw(ArgumentError("maximum must lie in (0, 1)"))

    k, tr, yr, maxval = promote(
        float(K),
        float(trend),
        float(reference_year),
        float(maximum),
    )

    return DoddDetection(k, tr, yr, maxval)
end

@inline function detection_probability(d::DoddDetection, t)
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


# ------------------------------------------------------------
# Dodd model parameters
# ------------------------------------------------------------

struct DoddTBParameters{T<:Real}
    Lambda0::T
    beta::T
    psi::T
    kappa::T

    epsilon::Vector{T}
    nu::Vector{T}
    omega::Vector{T}

    self_cure::Vector{T}
    tb_mortality::Vector{T}
    detection_multiplier::Vector{T}

    treatment_rate::T
    treatment_cfr::T

    detection::DoddDetection{T}
end


# ------------------------------------------------------------
# Parameter constructor
# ------------------------------------------------------------

function make_parameters(
    model::DoddTBModel;
    Lambda0::Real,
    beta::Real,
    psi::Real,
    kappa::Real,
    epsilon::AbstractVector,
    nu::AbstractVector,
    omega::AbstractVector,
    duration::AbstractVector,
    fatality::AbstractVector,
    relative_detection::AbstractVector,
    treatment_rate::Real=2.0,
    treatment_cfr::Real,
    detection::DoddDetection,
)
    nages = model.layout.nages

    for (name, values) in (
        (:epsilon, epsilon),
        (:nu, nu),
        (:omega, omega),
        (:duration, duration),
        (:fatality, fatality),
        (:relative_detection, relative_detection),
    )
        length(values) == nages ||
            throw(DimensionMismatch(
                "$name must have length $nages"
            ))
    end

    all(x -> isfinite(x) && x > 0, duration) ||
        throw(ArgumentError("Durations must be positive"))

    all(x -> isfinite(x) && 0 <= x <= 1, fatality) ||
        throw(ArgumentError("Fatality proportions must lie in [0, 1]"))

    all(x -> isfinite(x) && x >= 0,
        (Lambda0, beta, psi, kappa, treatment_rate)) ||
        throw(ArgumentError("Rates and multipliers must be nonnegative"))

    0 <= treatment_cfr <= 1 ||
        throw(ArgumentError("Treatment CFR must lie in [0, 1]"))

    for values in (epsilon, nu, omega, relative_detection)
        all(x -> isfinite(x) && x >= 0, values) ||
            throw(ArgumentError("Age-specific rates and multipliers must be nonnegative"))
    end

    # Promote all numerical inputs to a common type.
    # A differentiated parameter therefore propagates
    # its AD type through the complete parameter object.

    T = promote_type(
        Float64,
        typeof(Lambda0),
        typeof(beta),
        typeof(psi),
        typeof(kappa),
        eltype(epsilon),
        eltype(nu),
        eltype(omega),
        eltype(duration),
        eltype(fatality),
        eltype(relative_detection),
        typeof(treatment_rate),
        typeof(treatment_cfr),
        typeof(detection.K),
        typeof(detection.trend),
    )


    # Explicitly construct every field using the
    # promoted numerical type T.

    D = Vector{T}(duration)
    Φ = Vector{T}(fatality)

    self_cure = Vector{T}((one(T) .- Φ) ./ D)
    tb_mortality = Vector{T}(Φ ./ D)

    detection_multiplier = Vector{T}(
        Vector{T}(relative_detection) ./ D
    )

    typed_detection = DoddDetection{T}(
        T(detection.K),
        T(detection.trend),
        T(detection.reference_year),
        T(detection.maximum),
    )

    return DoddTBParameters{T}(
        T(Lambda0),
        T(beta),
        T(psi),
        T(kappa),
        Vector{T}(epsilon),
        Vector{T}(nu),
        Vector{T}(omega),
        self_cure,
        tb_mortality,
        detection_multiplier,
        T(treatment_rate),
        T(treatment_cfr),
        typed_detection,
    )
end



"""
    make_parameters(model::DoddTBModel, draw::NamedTuple)

Convert a draw of the Dodd model's free parameters into
the complete age-specific parameter structure.

The input parameters follow the naming conventions
established in priors(model).

Age-specific modifications follow Dodd et al. (2023),
Supplementary Methods, and the original implementation.
"""
function make_parameters(
    model::DoddTBModel,
    draw::NamedTuple,
)
    nages = state_layout(model).nages

    # --------------------------------------------------------
    # Age-specific progression rates
    # --------------------------------------------------------

    NumericT = promote_type(
        Float64,
        map(typeof, values(draw))...,
    )

    epsilon = NumericT[
        age < 5 ? draw.epsilon_child : draw.epsilon
        for age in 0:(nages-1)
    ]

    nu = fill(NumericT(draw.nu), nages)
    omega = fill(NumericT(draw.omega), nages)

    # --------------------------------------------------------
    # Untreated disease duration
    # --------------------------------------------------------

    # Dodd assumes a six-month untreated duration
    # for children younger than 15 years.

    duration = NumericT[
        age < 15 ? 0.5 : draw.duration
        for age in 0:(nages-1)
    ]

    fatality = fill(
        NumericT(draw.untreated_cfr), nages
    )

    # --------------------------------------------------------
    # Age-specific detection odds ratios
    # --------------------------------------------------------

    relative_detection = NumericT[
        age < 5 ? draw.detection_or_0_4 :
        age < 15 ? draw.detection_or_5_14 :
        1.0
        for age in 0:(nages-1)
    ]

    # --------------------------------------------------------
    # Detection trajectory
    # --------------------------------------------------------

    # Convert the sampled logit-scale parameter into
    # the logistic-scale K expected by DoddDetection.

    z = draw.detection_logitK

    K = if z >= 0
        inv(1 + exp(-z))
    else
        ez = exp(z)
        ez / (1 + ez)
    end

    detection = DoddDetection(
        K=K,
        trend=draw.detection_trend,
        reference_year=1970.0,
        maximum=0.9,
    )

    # --------------------------------------------------------
    # Construct deterministic parameters
    # --------------------------------------------------------

    return make_parameters(
        model;
        Lambda0=draw.Lambda0,
        beta=draw.beta,
        psi=draw.psi,
        kappa=draw.kappa,
        epsilon=epsilon,
        nu=nu,
        omega=omega,
        duration=duration,
        fatality=fatality,
        relative_detection=relative_detection,
        treatment_rate=2.0,
        treatment_cfr=draw.treatment_cfr,
        detection=detection,
    )
end



# ------------------------------------------------------------
# Detection hazard
# ------------------------------------------------------------

@inline function detection_rate(
    p::DoddTBParameters,
    age_index::Int,
    t,
)
    probability = detection_probability(p.detection, t)

    return p.detection_multiplier[age_index] *
        probability / (1 - probability)
end