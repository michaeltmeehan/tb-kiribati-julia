
"""
    quadrature_convergence(fit; orders=(30, 50, 70, 100))

Reevaluate the fitted marginal log-likelihood at
different Gauss-Hermite quadrature orders.

The epidemiological model is simulated only once.
Parameters are held fixed at their MAP estimates.
"""
function quadrature_convergence(
    fit::MAPFit;
    orders=(30, 50, 70, 100),
)
    obsmodel = fit.problem.observation_model

    obsmodel isa AnnualRandomEffectNB2Observation ||
        throw(ArgumentError(
            "Requires AnnualRandomEffectNB2Observation"
        ))

    all(n -> n >= 1, orders) ||
        throw(ArgumentError("Invalid quadrature order"))

    output = predict(fit)
    rows = notification_fit_rows(fit.problem, output)

    results = NamedTuple[]

    for n in orders
        model = AnnualRandomEffectNB2Observation(
            obsmodel.sigma_prior,
            obsmodel.tau_prior;
            quadrature_order=n,
        )

        ll = 0.0

        for obs in fit.problem.observations
            yearly = filter(r -> r.year == obs.year, rows)

            ll += observation_year_loglikelihood(
                model,
                yearly,
                fit.parameters,
            )
        end

        push!(results, (
            order=Int(n),
            log_likelihood=ll,
        ))
    end

    reference = last(results).log_likelihood

    return [
        (
            result...,
            difference=result.log_likelihood - reference,
        )
        for result in results
    ]
end


# ------------------------------------------------------------
# Conditional posterior distribution of annual effects
# ------------------------------------------------------------

"""
Posterior quadrature weights for a single year's
shared lognormal notification multiplier.

Weights are proportional to:

    quadrature_weight[k] *
    prod_g p(y_g | mu_g * multiplier[k])
"""
function _annual_multiplier_weights(
    rows,
    sigma,
    tau,
    nodes,
    weights,
)
    n = length(nodes)

    logweights = Vector{Float64}(undef, n)

    for k in 1:n
        multiplier = exp(
            tau * nodes[k] - tau^2 / 2
        )

        ll = log(weights[k])

        for row in rows
            ll += nb2_loglikelihood(
                row.observed,
                row.expected * multiplier,
                sigma,
            )
        end

        logweights[k] = ll
    end

    maximum_logweight = maximum(logweights)

    isfinite(maximum_logweight) ||
        throw(ArgumentError(
            "Annual-effect posterior has zero likelihood"
        ))

    probabilities = exp.(
        logweights .- maximum_logweight
    )

    probabilities ./= sum(probabilities)

    return probabilities
end


"""
Weighted quantile of values ordered increasingly.
"""
function _annual_weighted_quantile(
    values,
    probabilities,
    p,
)
    0 <= p <= 1 ||
        throw(ArgumentError("Quantile outside [0,1]"))

    index = searchsortedfirst(
        cumsum(probabilities),
        p,
    )

    return values[clamp(index, 1, length(values))]
end


"""
    posterior_annual_effects(
        fit;
        quadrature_order=100,
        level=0.95,
    )

Calculate the conditional posterior distribution of each
annual notification multiplier at the MAP parameter values.

Returns annual posterior means, equal-tailed intervals,
and expected notification counts conditional on the
posterior mean multiplier.
"""
function posterior_annual_effects(
    fit::MAPFit;
    quadrature_order::Int=100,
    level::Real=0.95,
)
    obsmodel = fit.problem.observation_model

    obsmodel isa AnnualRandomEffectNB2Observation ||
        throw(ArgumentError(
            "Requires AnnualRandomEffectNB2Observation"
        ))

    0 < level < 1 ||
        throw(ArgumentError("level must be between 0 and 1"))

    sigma = fit.parameters.sigma_obs
    tau = fit.parameters.tau_year

    nodes, weights =
        calibration_normal_quadrature(quadrature_order)

    multipliers = exp.(
        tau .* nodes .- tau^2 / 2
    )

    output = predict(fit)
    rows = notification_fit_rows(fit.problem, output)

    alpha = (1 - level) / 2

    results = NamedTuple[]

    for obs in fit.problem.observations
        yearly = filter(r -> r.year == obs.year, rows)

        observed = sum(r.observed for r in yearly)
        expected = sum(r.expected for r in yearly)

        if tau == 0
            mean_m = 1.0
            median_m = 1.0
            lower_m = 1.0
            upper_m = 1.0
        else
            probabilities = _annual_multiplier_weights(
                yearly,
                sigma,
                tau,
                nodes,
                weights,
            )

            mean_m = sum(
                probabilities .* multipliers
            )

            median_m = _annual_weighted_quantile(
                multipliers,
                probabilities,
                0.5,
            )

            lower_m = _annual_weighted_quantile(
                multipliers,
                probabilities,
                alpha,
            )

            upper_m = _annual_weighted_quantile(
                multipliers,
                probabilities,
                1 - alpha,
            )
        end

        push!(results, (
            year=obs.year,
            observed=observed,
            expected=expected,
            observed_ratio=expected > 0 ?
                           observed / expected : NaN,
            mean_multiplier=mean_m,
            median_multiplier=median_m,
            lower_multiplier=lower_m,
            upper_multiplier=upper_m,
            conditional_expected=expected * mean_m,
        ))
    end

    return results
end
