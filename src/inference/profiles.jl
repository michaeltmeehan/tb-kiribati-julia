
using Distributions: quantile

"""
    profile_likelihood(
        fit::MLEFit,
        parameter;
        grid,
        nstarts=2,
        iterations=3000,
    )

Profile the likelihood for one parameter.

Every grid point is fitted by fixing the target
parameter and reoptimising all other free parameters.

The global MLE is used as a reference and warm start.
"""
function profile_likelihood(
    fit::MLEFit,
    parameter::Symbol;
    grid,
    optimizer::Symbol=:nelder_mead,
    nstarts::Int=2,
    iterations::Int=3000,
    rng::AbstractRNG=Random.default_rng(),
)
    problem = fit.problem

    haskey(problem.prior, parameter) ||
        throw(ArgumentError(
            "Unknown parameter: $parameter"
        ))

    haskey(fit.fixed, parameter) &&
        throw(ArgumentError(
            "Cannot profile an already fixed parameter"
        ))

    values = sort(unique(Float64.(collect(grid))))

    isempty(values) &&
        throw(ArgumentError("Profile grid is empty"))

    mle = Float64(getproperty(
        fit.parameters, parameter
    ))

    results = NamedTuple[]

    # Profile outward from the fitted value.
    left = reverse(filter(x -> x < mle, values))
    right = filter(x -> x >= mle, values)

    for branch in (left, right)
        initial = fit.parameters

        for value in branch
            fixed = merge(
                fit.fixed,
                NamedTuple{(parameter,)}((value,)),
            )

            candidate = fit_mle(
                problem;
                optimizer=optimizer,
                fixed=fixed,
                initial=initial,
                nstarts=nstarts,
                iterations=iterations,
                rng=rng,
            )

            push!(results, (
                value=value,
                log_likelihood=candidate.log_likelihood,
                converged=candidate.converged,
                parameters=candidate.parameters,
            ))

            initial = candidate.parameters
        end
    end

    sort!(results, by=r -> r.value)

    # Preserve the original reference likelihood.
    # Negative deviances indicate a better solution was
    # found during profiling, requiring a new global fit.
    points = [
        (
            result...,
            deviance=2 * (
                fit.log_likelihood -
                    result.log_likelihood
            ),
        )
        for result in results
    ]

    return (
        parameter=parameter,
        mle=mle,
        reference_loglikelihood=fit.log_likelihood,
        points=points,
    )
end


"""
    profile_likelihoods(
        fit;
        parameters,
        probabilities,
        ...
    )

Generate initial profiles for multiple parameters.
The grids are based on prior quantiles solely as
starting exploration ranges.

These are not confidence intervals or hard limits.
"""
function profile_likelihoods(
    fit::MLEFit;
    parameters=collect(keys(fit.problem.prior)),
    probabilities=(
        0.001, 0.005, 0.025, 0.1, 0.25,
        0.5, 0.75, 0.9, 0.975, 0.995, 0.999
    ),
    kwargs...,
)
    results = Dict{Symbol,Any}()

    for parameter in parameters
        name = Symbol(parameter)

        haskey(fit.fixed, name) && continue

        dist = getproperty(fit.problem.prior, name)
        mle = getproperty(fit.parameters, name)

        grid = sort(unique(vcat(
            [quantile(dist, p) for p in probabilities],
            [mle],
        )))

        results[name] = profile_likelihood(
            fit,
            name;
            grid=grid,
            kwargs...,
        )
    end

    return results
end
