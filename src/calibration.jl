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
and return true annual incidence aggregated into the requested age groups.

`model_parameters` contains the parameters varied by the calibration or
uncertainty analysis. `fixed_parameters` contains any non-default parameters
that should remain fixed.
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
# Notification likelihood
# ---------------------------------------------------------------------------

"""
    notification_loglikelihood(
        predicted_incidence,
        notification_observations,
    )

Evaluate the multinomial likelihood for observed notification age
compositions.

Each element of `notification_observations` must contain:

    year
    counts
    design
    labels

where `design * predicted_incidence` gives the expected relative notification
weight for each observed age category.
"""
function notification_loglikelihood(
    predicted_incidence,
    notification_observations,
)

    loglik = 0.0

    for observation in notification_observations

        weights =
            observation.design * predicted_incidence

        if any(x -> !isfinite(x) || x <= 0, weights)
            return -Inf
        end

        probabilities = weights ./ sum(weights)
        total_cases = sum(observation.counts)

        loglik += logpdf(
            Multinomial(total_cases, probabilities),
            observation.counts,
        )
    end

    return loglik
end


# ---------------------------------------------------------------------------
# WHO incidence likelihood
# ---------------------------------------------------------------------------

"""
    incidence_loglikelihood(
        predicted_incidence,
        incidence_observations,
    )

Evaluate the likelihood for WHO incidence estimates.

Each observation must contain:

    year
    observed
    lower
    upper

The reported uncertainty interval is approximated by a log-normal
distribution, with the reported central estimate treated as the median.
"""
function incidence_loglikelihood(
    predicted_incidence,
    incidence_observations,
)

    predicted_rate =
        model_total_incidence_rate(predicted_incidence)

    if !isfinite(predicted_rate) || predicted_rate <= 0
        return -Inf
    end

    loglik = 0.0

    for observation in incidence_observations

        observed = observation.observed
        lower = observation.lower
        upper = observation.upper

        if observed <= 0 || lower <= 0 || upper <= 0
            continue
        end

        sigma =
            (log(upper) - log(lower)) /
            (2 * 1.96)

        loglik += logpdf(
            LogNormal(log(predicted_rate), sigma),
            observed,
        )
    end

    return loglik
end


# ---------------------------------------------------------------------------
# Combined calibration likelihood
# ---------------------------------------------------------------------------

"""
    calibration_loglikelihood(
        model_parameters;
        notification_observations,
        incidence_observations,
        age_breaks,
        fixed_parameters = NamedTuple(),
        tinit = 1800.0,
        tfinal = 2024.0,
    )

Run the transmission model and evaluate the combined calibration likelihood.
"""
function calibration_loglikelihood(
    model_parameters::NamedTuple;
    notification_observations,
    incidence_observations,
    age_breaks,
    fixed_parameters::NamedTuple = NamedTuple(),
    tinit::Real = 1800.0,
    tfinal::Real = 2024.0,
)

    predicted_incidence =
        model_incidence_by_age_group(
            model_parameters;
            age_breaks = age_breaks,
            fixed_parameters = fixed_parameters,
            tinit = tinit,
            tfinal = tfinal,
        )

    if any(x -> !isfinite(x) || x <= 0, predicted_incidence)
        return -Inf
    end

    notification_ll =
        notification_loglikelihood(
            predicted_incidence,
            notification_observations,
        )

    incidence_ll =
        incidence_loglikelihood(
            predicted_incidence,
            incidence_observations,
        )

    if !isfinite(notification_ll) ||
       !isfinite(incidence_ll)

        return -Inf
    end

    return notification_ll + incidence_ll
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