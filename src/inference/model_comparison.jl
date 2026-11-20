function _check_comparable_observations(reduced, full)
    a = reduced.problem
    b = full.problem

    a.observations == b.observations ||
        throw(ArgumentError(
            "Fits use different notification observations"
        ))

    isequal(a.treatment_deaths, b.treatment_deaths) ||
        throw(ArgumentError(
            "Fits use different treatment-death observations"
        ))

    isequal(a.recurrence, b.recurrence) ||
        throw(ArgumentError(
            "Fits use different recurrence observations"
        ))

    a.age_breaks == b.age_breaks ||
        throw(ArgumentError(
            "Fits use different age groupings"
        ))

    return nothing
end

"""
    compare_calibrations(fits)

Summarise maximum-likelihood fits.

Accepts a NamedTuple of MLEFit objects.
"""
function compare_calibrations(fits::NamedTuple)
    all(fit -> fit isa MLEFit, values(fits)) ||
        throw(ArgumentError(
            "Model comparison requires MLEFit objects"
        ))

    return [
        (
            model=String(name),
            log_likelihood=fit.log_likelihood,
            n_parameters=fit.n_parameters,
            aic=-2 * fit.log_likelihood +
                2 * fit.n_parameters,
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
    _check_comparable_observations(reduced, full)

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
