# ------------------------------------------------------------
# Detection trajectory
# ------------------------------------------------------------

struct DoddDetection
    K::Float64
    trend::Float64
    reference_year::Float64
    maximum::Float64
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

    return DoddDetection(
        Float64(K),
        Float64(trend),
        Float64(reference_year),
        Float64(maximum),
    )
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

struct DoddTBParameters
    Lambda0::Float64
    beta::Float64
    psi::Float64
    kappa::Float64

    epsilon::Vector{Float64}
    nu::Vector{Float64}
    omega::Vector{Float64}

    self_cure::Vector{Float64}
    tb_mortality::Vector{Float64}
    detection_multiplier::Vector{Float64}

    treatment_rate::Float64
    treatment_cfr::Float64

    detection::DoddDetection
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

    D = Float64.(duration)
    Φ = Float64.(fatality)

    self_cure = (1 .- Φ) ./ D
    tb_mortality = Φ ./ D
    detection_multiplier = Float64.(relative_detection) ./ D

    return DoddTBParameters(
        Float64(Lambda0),
        Float64(beta),
        Float64(psi),
        Float64(kappa),
        Float64.(epsilon),
        Float64.(nu),
        Float64.(omega),
        self_cure,
        tb_mortality,
        detection_multiplier,
        Float64(treatment_rate),
        Float64(treatment_cfr),
        detection,
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

    epsilon = [
        age < 5 ? draw.epsilon_child : draw.epsilon
        for age in 0:(nages-1)
    ]

    nu = fill(draw.nu, nages)
    omega = fill(draw.omega, nages)

    # --------------------------------------------------------
    # Untreated disease duration
    # --------------------------------------------------------

    # Dodd assumes a six-month untreated duration
    # for children younger than 15 years.

    duration = [
        age < 15 ? 0.5 : draw.duration
        for age in 0:(nages-1)
    ]

    fatality = fill(draw.untreated_cfr, nages)

    # --------------------------------------------------------
    # Age-specific detection odds ratios
    # --------------------------------------------------------

    relative_detection = [
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