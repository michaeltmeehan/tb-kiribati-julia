using TBKiribatiJulia

using Optim

include("calibration_data.jl")


# ---------------------------------------------------------------------------
# Parameters included in this calibration
# ---------------------------------------------------------------------------

const MODEL_PARAMETERS = (
    :beta,
    :progression_child,
    :progression_5_14,
    :progression_15_64,
    :progression_65_plus,
)

const N_MODEL_PARAMETERS = length(MODEL_PARAMETERS)


# ---------------------------------------------------------------------------
# Model and likelihood wrappers
# ---------------------------------------------------------------------------

function model_parameter_namedtuple(x)

    length(x) >= N_MODEL_PARAMETERS ||
        error("Parameter vector has incorrect length")

    return NamedTuple{MODEL_PARAMETERS}(
        Tuple(x[1:N_MODEL_PARAMETERS]),
    )
end


function unpack_parameters(x)

    length(x) == N_MODEL_PARAMETERS + 1 ||
        error("Parameter vector has incorrect length")

    model_parameters = model_parameter_namedtuple(x)
    sigma_obs = x[end]

    return model_parameters, sigma_obs
end


function model_predictions(model_parameters)

    return model_incidence_by_age_group(
        model_parameters;
        age_breaks = CALIBRATION_AGE_BREAKS,
        fixed_parameters = FIXED_PARAMETERS,
        tinit = TINIT,
        tfinal = TFINAL,
    )
end


function loglikelihood(model_parameters, sigma_obs)

    return calibration_loglikelihood(
        model_parameters;
        sigma_obs = sigma_obs,
        notification_observations = NOTIFICATION_OBSERVATIONS,
        relative_detection = RELATIVE_DETECTION,
        age_breaks = CALIBRATION_AGE_BREAKS,
        fixed_parameters = FIXED_PARAMETERS,
        tinit = TINIT,
        tfinal = TFINAL,
    )
end


function objective(x)

    model_parameters, sigma_obs = unpack_parameters(x)
    ll = loglikelihood(model_parameters, sigma_obs)

    return isfinite(ll) ? -ll : Inf
end


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

# Transmission starting values are close to the centres of the current priors.
# sigma_obs = 0.1 corresponds to phi = 100.

x0 = [
    1.0,    # beta
    1.5,    # progression_child
    0.1,    # progression_5_14
    0.2,    # progression_15_64
    0.4,    # progression_65_plus
    0.1,    # sigma_obs
]


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

lower = [
    0.01,    # beta
    0.01,    # progression_child
    0.01,    # progression_5_14
    0.01,    # progression_15_64
    0.01,    # progression_65_plus
    1.0e-4,  # sigma_obs; small positive value approximates Poisson limit
]

upper = [
     5.0,    # beta
    10.0,    # progression_child
    10.0,    # progression_5_14
    10.0,    # progression_15_64
    10.0,    # progression_65_plus
     2.0,    # sigma_obs
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
        iterations = 1000,
        show_trace = true,
        show_every = 20,
    ),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

xhat = Optim.minimizer(result)
fitted_parameters, fitted_sigma_obs = unpack_parameters(xhat)
fitted_phi = inv(fitted_sigma_obs^2)

println()
println("Converged: ", Optim.converged(result))
println("Minimum negative log-likelihood: ", Optim.minimum(result))
println()
println("Fitted parameters:")

for (parameter, value) in pairs(fitted_parameters)
    println(
        rpad(string(parameter), 25),
        " = ",
        value,
    )
end

println(
    rpad("sigma_obs", 25),
    " = ",
    fitted_sigma_obs,
)

println(
    rpad("phi", 25),
    " = ",
    fitted_phi,
)


# ---------------------------------------------------------------------------
# Fitted incidence
# ---------------------------------------------------------------------------

fitted_incidence = model_predictions(fitted_parameters)
fitted_total_rate = model_total_incidence_rate(fitted_incidence)

population = get_population(EQUILIBRIUM_YEAR)
population_by_group = aggregate_age_groups(
    population,
    CALIBRATION_AGE_BREAKS,
)

fitted_incidence_rates =
    1e5 .* fitted_incidence ./ population_by_group

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
        "  ",
        CALIBRATION_AGE_LABELS[g],
        ": cases = ",
        round(fitted_incidence[g], digits = 2),
        ", rate = ",
        round(fitted_incidence_rates[g], digits = 2),
        " per 100,000",
    )
end


# ---------------------------------------------------------------------------
# Notification-count diagnostics
# ---------------------------------------------------------------------------

println()
println("Notification count fit:")

for observation in NOTIFICATION_OBSERVATIONS

    detection = age_specific_detection(
        fitted_incidence,
        observation.overall_detection,
        RELATIVE_DETECTION,
    )

    expected = expected_notification_counts(
        fitted_incidence,
        observation;
        relative_detection = RELATIVE_DETECTION,
    )

    isnothing(detection) &&
        error("Invalid fitted detection probabilities for $(observation.year)")

    isnothing(expected) &&
        error("Invalid fitted expected counts for $(observation.year)")

    println()
    println(
        observation.year,
        "  overall detection = ",
        round(observation.overall_detection, digits = 3),
        "  observed total = ",
        sum(observation.counts),
        "  expected total = ",
        round(sum(expected), digits = 1),
    )

    for i in eachindex(observation.labels)
        println(
            "  ",
            rpad(observation.labels[i], 6),
            ": observed = ",
            lpad(observation.counts[i], 4),
            ", expected = ",
            lpad(round(expected[i], digits = 1), 6),
        )
    end
end


# ---------------------------------------------------------------------------
# Detection probabilities
# ---------------------------------------------------------------------------

println()
println("Age-specific detection probabilities:")

for observation in NOTIFICATION_OBSERVATIONS

    detection = age_specific_detection(
        fitted_incidence,
        observation.overall_detection,
        RELATIVE_DETECTION,
    )

    isnothing(detection) &&
        error("Invalid fitted detection probabilities for $(observation.year)")

    println()
    println(
        observation.year,
        "  overall = ",
        round(observation.overall_detection, digits = 3),
    )

    for g in eachindex(CALIBRATION_AGE_LABELS)
        println(
            "  ",
            rpad(CALIBRATION_AGE_LABELS[g], 6),
            " = ",
            round(detection[g], digits = 3),
        )
    end
end
