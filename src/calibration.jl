using Distributions


# ---------------------------------------------------------------------------
# Model-predicted incidence
# ---------------------------------------------------------------------------

"""
    model_incidence_by_age_group(
        model_parameters;
        age_breaks,
        fixed_parameters = NamedTuple(),
        tinit = 1800.0,
        tfinal = 2024.0,
    )

Run the equilibrium-demography TB model for a supplied set of model parameters
and return true annual incident case counts aggregated into the requested age
groups.
"""
function model_incidence_by_age_group(
    model_parameters::NamedTuple;
    age_breaks,
    fixed_parameters::NamedTuple = NamedTuple(),
    tinit::Real = 1800.0,
    tfinal::Real = 2024.0,
)
    parameter_kwargs = merge(fixed_parameters, model_parameters)

    params = make_parameters(
        CONTACT;
        parameter_kwargs...,
    )

    population = get_population(EQUILIBRIUM_YEAR)
    u0 = seeded_initial_state(population)
    times = Float64(tinit):1.0:Float64(tfinal)

    sol = simulate(
        params;
        tspan = (Float64(tinit), Float64(tfinal)),
        u0 = u0,
        saveat = times,
        demography = :equilibrium,
    )

    _, incidence_by_age = raw_annual_incidence_by_age(sol)

    return aggregate_age_groups(
        incidence_by_age[end, :],
        age_breaks,
    )
end


"""
    model_total_incidence_rate(predicted_incidence)

Convert annual incident case counts into an incidence rate per 100,000 using
the equilibrium-year population.
"""
function model_total_incidence_rate(predicted_incidence)
    population = get_population(EQUILIBRIUM_YEAR)
    return 1e5 * sum(predicted_incidence) / sum(population)
end


# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------

"""
    age_specific_detection(
        predicted_incidence,
        overall_detection,
        relative_detection,
    )

Construct age-specific detection probabilities while preserving the supplied
overall (incidence-weighted) detection fraction exactly.

`relative_detection` specifies only the relative age pattern. A vector of ones
therefore gives the same detection probability in every age group.

Returns `nothing` if the requested relative pattern implies an invalid
age-specific detection probability outside (0, 1].
"""
function age_specific_detection(
    predicted_incidence,
    overall_detection::Real,
    relative_detection,
)
    if !isfinite(overall_detection) ||
       overall_detection <= 0 ||
       overall_detection > 1
        return nothing
    end

    if length(relative_detection) != length(predicted_incidence) ||
       any(x -> !isfinite(x) || x <= 0, relative_detection)
        return nothing
    end

    denominator = sum(relative_detection .* predicted_incidence)
    total_incidence = sum(predicted_incidence)

    if !isfinite(denominator) || denominator <= 0 ||
       !isfinite(total_incidence) || total_incidence <= 0
        return nothing
    end

    scale = overall_detection * total_incidence / denominator
    detection = scale .* relative_detection

    if any(x -> !isfinite(x) || x <= 0 || x > 1, detection)
        return nothing
    end

    return detection
end


"""
    expected_notification_counts(
        predicted_incidence,
        observation;
        relative_detection,
    )

Calculate expected notification counts in the observed age categories for one
year. Detection is applied to the model age groups first; `observation.design`
then performs only age-category aggregation.
"""
function expected_notification_counts(
    predicted_incidence,
    observation;
    relative_detection,
)
    detection = age_specific_detection(
        predicted_incidence,
        observation.overall_detection,
        relative_detection,
    )

    isnothing(detection) && return nothing

    detected_incidence = detection .* predicted_incidence
    expected = observation.design * detected_incidence

    if any(x -> !isfinite(x) || x <= 0, expected)
        return nothing
    end

    return expected
end


# ---------------------------------------------------------------------------
# Negative-binomial notification likelihood
# ---------------------------------------------------------------------------

"""
    negative_binomial_mu_phi(mu, phi)

Construct an NB2 distribution with mean `mu` and dispersion `phi`, such that

    Var(Y) = mu + mu^2 / phi.

Julia's `NegativeBinomial(r, p)` uses `r = phi` and
`p = phi / (phi + mu)` under this parameterisation.
"""
function negative_binomial_mu_phi(
    mu::Real,
    phi::Real,
)
    if !isfinite(mu) || mu <= 0 ||
       !isfinite(phi) || phi <= 0
        return nothing
    end

    p = phi / (phi + mu)
    return NegativeBinomial(phi, p)
end


"""
    notification_loglikelihood(
        predicted_incidence,
        notification_observations;
        sigma_obs,
        relative_detection,
    )

Evaluate the negative-binomial likelihood for observed age-specific
notification counts. `sigma_obs` is the extra-Poisson coefficient of variation
parameter, with `phi = 1 / sigma_obs^2`.
"""
function notification_loglikelihood(
    predicted_incidence,
    notification_observations;
    sigma_obs::Real,
    relative_detection,
)
    if !isfinite(sigma_obs) || sigma_obs <= 0
        return -Inf
    end

    phi = inv(sigma_obs^2)
    loglik = 0.0

    for observation in notification_observations
        expected = expected_notification_counts(
            predicted_incidence,
            observation;
            relative_detection = relative_detection,
        )

        isnothing(expected) && return -Inf

        for (observed, mu) in zip(observation.counts, expected)
            dist = negative_binomial_mu_phi(mu, phi)
            isnothing(dist) && return -Inf
            loglik += logpdf(dist, observed)
        end
    end

    return loglik
end


# ---------------------------------------------------------------------------
# Combined calibration likelihood
# ---------------------------------------------------------------------------

"""
    calibration_loglikelihood(
        model_parameters;
        sigma_obs,
        notification_observations,
        relative_detection,
        age_breaks,
        fixed_parameters = NamedTuple(),
        tinit = 1800.0,
        tfinal = 2024.0,
    )

Run the transmission model and evaluate the negative-binomial notification
likelihood. WHO incidence estimates enter only through the fixed annual overall
detection fractions stored in `notification_observations`.
"""
function calibration_loglikelihood(
    model_parameters::NamedTuple;
    sigma_obs::Real,
    notification_observations,
    relative_detection,
    age_breaks,
    fixed_parameters::NamedTuple = NamedTuple(),
    tinit::Real = 1800.0,
    tfinal::Real = 2024.0,
)
    predicted_incidence = model_incidence_by_age_group(
        model_parameters;
        age_breaks = age_breaks,
        fixed_parameters = fixed_parameters,
        tinit = tinit,
        tfinal = tfinal,
    )

    if any(x -> !isfinite(x) || x <= 0, predicted_incidence)
        return -Inf
    end

    return notification_loglikelihood(
        predicted_incidence,
        notification_observations;
        sigma_obs = sigma_obs,
        relative_detection = relative_detection,
    )
end


"""
    calibration_loss(args...; kwargs...)

Negative calibration log-likelihood, suitable for minimisation with Optim.
"""
function calibration_loss(
    model_parameters::NamedTuple;
    kwargs...,
)
    return -calibration_loglikelihood(
        model_parameters;
        kwargs...,
    )
end
