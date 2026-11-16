
"""
    compare_calibrations(fits)

Summarise maximum-likelihood fits.

Accepts a NamedTuple of MLEFit objects.
"""
function compare_calibrations(fits::NamedTuple)
    return [
        (
            model=String(name),
            log_likelihood=fit.log_likelihood,
            n_parameters=fit.n_parameters,
            aic=-2fit.log_likelihood +
                2fit.n_parameters,
            converged=fit.converged,
        )
        for (name, fit) in pairs(fits)
    ]
end


"""
    likelihood_ratio(reduced, full)

Calculate a likelihood-ratio statistic for two
nested MLE models.

The caller is responsible for verifying nesting
and selecting the appropriate null distribution.
"""
function likelihood_ratio(
    reduced::MLEFit,
    full::MLEFit,
)
    reduced.problem.observations ==
        full.problem.observations ||
        throw(ArgumentError(
            "Fits must use the same observations"
        ))

    df = full.n_parameters - reduced.n_parameters

    df > 0 ||
        throw(ArgumentError(
            "Full model must have more parameters"
        ))

    statistic = 2 * (
        full.log_likelihood -
            reduced.log_likelihood
    )

    return (
        statistic=statistic,
        df=df,
        loglikelihood_difference=
        full.log_likelihood -
            reduced.log_likelihood,
        reduced_converged=reduced.converged,
        full_converged=full.converged,
    )
end
