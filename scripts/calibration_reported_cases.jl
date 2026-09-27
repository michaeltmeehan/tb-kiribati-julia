using TBKiribatiJulia

using DataFrames
using Optim
using Statistics

include("calibration_data.jl")


# ---------------------------------------------------------------------------
# Parameters included in this calibration
# ---------------------------------------------------------------------------

const CALIBRATION_PARAMETERS = (
    :beta,
    :progression_child,
    :progression_5_14,
    :progression_15_64,
    :progression_65_plus,
)


# ---------------------------------------------------------------------------
# Model and likelihood wrappers
# ---------------------------------------------------------------------------

function model_predictions(model_parameters)

    return model_incidence_by_age_group(
        model_parameters;
        age_breaks = CALIBRATION_AGE_BREAKS,
        fixed_parameters = FIXED_PARAMETERS,
        tinit = TINIT,
        tfinal = TFINAL,
    )
end


function loglikelihood(model_parameters)

    return calibration_loglikelihood(
        model_parameters;
        notification_observations =
            NOTIFICATION_OBSERVATIONS,
        incidence_observations =
            INCIDENCE_OBSERVATIONS,
        age_breaks =
            CALIBRATION_AGE_BREAKS,
        fixed_parameters =
            FIXED_PARAMETERS,
        tinit =
            TINIT,
        tfinal =
            TFINAL,
    )
end


function loss(model_parameters)

    return -loglikelihood(model_parameters)
end


# ---------------------------------------------------------------------------
# Optim interface
# ---------------------------------------------------------------------------

function parameter_namedtuple(x)

    length(x) == length(CALIBRATION_PARAMETERS) ||
        error("Parameter vector has incorrect length")

    return NamedTuple{
        CALIBRATION_PARAMETERS
    }(Tuple(x))
end


function objective(x)

    return loss(
        parameter_namedtuple(x),
    )
end


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

x0 = [
    0.75,   # beta
    2.4,    # progression_child
    0.05,   # progression_5_14
    0.1,    # progression_15_64
    2.4,    # progression_65_plus
]


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

lower = [
    0.01,   # beta
    0.01,   # progression_child
    0.01,   # progression_5_14
    0.01,   # progression_15_64
    0.01,   # progression_65_plus
]

upper = [
     5.0,   # beta
    10.0,   # progression_child
    10.0,   # progression_5_14
    10.0,   # progression_15_64
    10.0,   # progression_65_plus
]


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

result = optimize(
    objective,
    lower,
    upper,
    x0,
    Fminbox(NelderMead()),
    Optim.Options(
        iterations = 500,
        show_trace = true,
        show_every = 20,
    ),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

xhat =
    Optim.minimizer(result)

fitted_parameters =
    parameter_namedtuple(xhat)


println()

println(
    "Converged: ",
    Optim.converged(result),
)

println(
    "Minimum negative log-likelihood: ",
    Optim.minimum(result),
)

println()

println("Fitted parameters:")

for (parameter, value) in pairs(fitted_parameters)

    println(
        rpad(string(parameter), 25),
        " = ",
        value,
    )
end


# ---------------------------------------------------------------------------
# Fitted incidence
# ---------------------------------------------------------------------------

fitted_incidence =
    model_predictions(fitted_parameters)

fitted_total_rate =
    model_total_incidence_rate(fitted_incidence)


println()

println(
    "Fitted equilibrium total incidence rate = ",
    round(fitted_total_rate, digits = 2),
    " per 100,000",
)

println()
println("Fitted true incidence by age group:")

for g in eachindex(CALIBRATION_AGE_LABELS)

    println(
        CALIBRATION_AGE_LABELS[g],
        ": ",
        round(fitted_incidence[g], digits = 2),
    )
end


# ---------------------------------------------------------------------------
# Implied absolute detection probabilities
# ---------------------------------------------------------------------------

infant_incidence =
    fitted_incidence[1]

child_incidence =
    fitted_incidence[2]

adult_incidence =
    sum(fitted_incidence[3:end])

total_incidence =
    infant_incidence +
    child_incidence +
    adult_incidence


implied_detection = DataFrame(
    year = Int[],
    overall = Float64[],
    infant = Float64[],
    child = Float64[],
    adult = Float64[],
)


for row in eachrow(INCIDENCE)

    year = Int(row.year)

    year in CALIBRATION_YEARS ||
        continue

    q_overall =
        Float64(row.detection_fraction)

    q_adult =
        q_overall * total_incidence /
        (
            INFANT_TO_ADULT_DETECTION_RATIO *
            infant_incidence +

            CHILD_TO_ADULT_DETECTION_RATIO *
            child_incidence +

            adult_incidence
        )

    q_infant =
        INFANT_TO_ADULT_DETECTION_RATIO *
        q_adult

    q_child =
        CHILD_TO_ADULT_DETECTION_RATIO *
        q_adult

    push!(
        implied_detection,
        (
            year = year,
            overall = q_overall,
            infant = q_infant,
            child = q_child,
            adult = q_adult,
        ),
    )
end


println()
println("Implied detection probabilities:")

for row in eachrow(implied_detection)

    println(
        row.year,
        ": overall = ",
        round(row.overall, digits = 3),

        ", 0-4 = ",
        round(row.infant, digits = 3),

        ", 5-14 = ",
        round(row.child, digits = 3),

        ", >=15 = ",
        round(row.adult, digits = 3),
    )
end


println()

println(
    "Mean implied detection: 0-4 = ",
    round(
        mean(implied_detection.infant),
        digits = 3,
    ),

    ", 5-14 = ",
    round(
        mean(implied_detection.child),
        digits = 3,
    ),

    ", >=15 = ",
    round(
        mean(implied_detection.adult),
        digits = 3,
    ),
)


# ---------------------------------------------------------------------------
# Age-composition diagnostics
# ---------------------------------------------------------------------------

println()
println("Notification age-composition fit:")


for observation in NOTIFICATION_OBSERVATIONS

    weights =
        observation.design *
        fitted_incidence

    probabilities =
        weights ./ sum(weights)

    expected =
        sum(observation.counts) .*
        probabilities


    println()
    println(observation.year)


    for i in eachindex(observation.labels)

        println(
            "  ",
            observation.labels[i],

            ": observed = ",
            observation.counts[i],

            ", expected = ",
            round(
                expected[i],
                digits = 1,
            ),
        )
    end
end