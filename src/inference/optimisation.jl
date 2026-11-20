
using Random
using Distributions: median, quantile
using Optim
using ADTypes: AutoForwardDiff, AutoFiniteDiff
using LineSearches

abstract type AbstractCalibrationFit end

struct MAPFit{P,T,R} <: AbstractCalibrationFit
    problem::P
    parameters::T
    log_prior::Float64
    log_likelihood::Float64
    log_posterior::Float64
    converged::Bool
    optim_result::R
    runs::Vector{Any}
end

struct MLEFit{P,T,F,R} <: AbstractCalibrationFit
    problem::P
    parameters::T
    fixed::F
    log_likelihood::Float64
    converged::Bool
    optim_result::R
    runs::Vector{Any}
    n_parameters::Int
end

# ------------------------------------------------------------
# Parameter transformations
# ------------------------------------------------------------

# Transform a parameter from its natural support to R.
function _to_unconstrained(x, dist)
    lo = minimum(dist)
    hi = maximum(dist)

    if isfinite(lo) && isfinite(hi)
        lo < x < hi ||
            throw(ArgumentError(
                "Initial value outside open parameter support"
            ))
        return log((x - lo) / (hi - x))

    elseif isfinite(lo)
        x > lo || throw(ArgumentError(
            "Initial value must exceed $lo"
        ))
        return log(x - lo)

    elseif isfinite(hi)
        x < hi || throw(ArgumentError(
            "Initial value must be below $hi"
        ))
        return log(hi - x)

    else
        return Float64(x)
    end
end

function _from_unconstrained(z, dist)
    lo = minimum(dist)
    hi = maximum(dist)

    if isfinite(lo) && isfinite(hi)
        # Stable logistic transformation.
        s = z >= 0 ?
            inv(1 + exp(-z)) :
            exp(z) / (1 + exp(z))

        return lo + (hi - lo) * s

    elseif isfinite(lo)
        return lo + exp(z)

    elseif isfinite(hi)
        return hi - exp(z)

    else
        return z
    end
end

# ------------------------------------------------------------
# Parameter assembly
# ------------------------------------------------------------

function _complete_parameters(
    prior::NamedTuple,
    fixed::NamedTuple,
    free::NamedTuple,
)
    # Preserve the ordering of the original calibration prior.
    names = keys(prior)

    return NamedTuple{names}(
        Tuple(
            haskey(fixed, name) ?
            getproperty(fixed, name) :
            getproperty(free, name)
            for name in names
        )
    )
end

function _free_priors(
    prior::NamedTuple,
    fixed::NamedTuple,
)
    all(k -> haskey(prior, k), keys(fixed)) ||
        throw(ArgumentError("Unknown fixed parameter"))

    names = Tuple(
        k for k in keys(prior) if !haskey(fixed, k)
    )

    isempty(names) &&
        throw(ArgumentError(
            "At least one parameter must remain free"
        ))

    return NamedTuple{names}(
        Tuple(getproperty(prior, k) for k in names)
    )
end

# ------------------------------------------------------------
# Unified optimisation
# ------------------------------------------------------------

function _optimise(objective, z0, optimizer, options)
    if optimizer == :nelder_mead
        return Optim.optimize(
            objective, z0, Optim.NelderMead(), options
        )
    end

    backend = if optimizer == :lbfgs
        AutoForwardDiff()
    elseif optimizer == :lbfgs_fd
        AutoFiniteDiff()
    else
        throw(ArgumentError("Unknown optimiser: $optimizer"))
    end

    return Optim.optimize(
        objective,
        z0,
        Optim.LBFGS(
            linesearch=LineSearches.BackTracking(),
        ),
        options;
        autodiff=backend,
    )
end

function _invalid_calibration_trial(err)
    if err isa ErrorException
        return startswith(err.msg, "ODE simulation failed:")
    end

    if err isa ArgumentError
        message = sprint(showerror, err)

        return any(
            text -> occursin(text, message),
            (
                "K must lie in (0, 1)",
                "Initialisation produces a negative latent population",
                "Negative epidemiological population",
            ),
        )
    end

    return false
end

function _fit_calibration(
    problem::CalibrationProblem;
    method::Symbol,
    optimizer::Symbol=:nelder_mead,
    fixed::NamedTuple=NamedTuple(),
    initial=nothing,
    rng::AbstractRNG=Random.default_rng(),
    nstarts::Int=3,
    iterations::Int=3000,
    show_trace::Bool=false,
)
    method in (:map, :mle) ||
        throw(ArgumentError("Unknown fitting method"))

    optimizer in (
        :nelder_mead,
        :lbfgs,
        :lbfgs_fd,
    ) || throw(ArgumentError("Unknown optimiser"))

    nstarts >= 1 ||
        throw(ArgumentError("nstarts must be positive"))

    iterations >= 1 ||
        throw(ArgumentError("iterations must be positive"))

    all(k -> haskey(problem.prior, k), keys(fixed)) ||
        throw(ArgumentError("Unknown fixed parameter"))

    free_priors = _free_priors(problem.prior, fixed)
    names = keys(free_priors)
    distributions = values(free_priors)

    function unpack(z)
        free = NamedTuple{names}(
            ntuple(
                j -> _from_unconstrained(
                    z[j], distributions[j]
                ),
                length(names),
            )
        )

        return _complete_parameters(
            problem.prior, fixed, free
        )
    end

    function pack(theta)
        return [
            _to_unconstrained(
                getproperty(theta, name),
                getproperty(free_priors, name),
            )
            for name in names
        ]
    end


    function objective(z)
        # Reject nonfinite transformed parameters.
        all(isfinite, z) || return 1e100

        theta = unpack(z)

        # Reject nonfinite natural parameters.
        all(isfinite, Base.values(theta)) || return 1e100

        value = try
            if method == :map
                calibration_logposterior(problem, theta)
            else
                calibration_loglikelihood(problem, theta)
            end

        catch err
            _invalid_calibration_trial(err) || rethrow()
            return 1e100
        end

        return isfinite(value) ? -value : 1e100
    end

    # Prior-based starts are a numerical convenience only.
    # The MLE objective never evaluates the prior density.
    starting_points = Any[]

    if !isnothing(initial)
        push!(starting_points, pack(initial))
    else
        centre = map(median, free_priors)
        push!(starting_points, pack(centre))
    end

    for _ in 2:nstarts
        draw = map(d -> rand(rng, d), free_priors)
        push!(starting_points, pack(draw))
    end

    runs = Any[]
    best = nothing

    options = Optim.Options(
        iterations=iterations,
        show_trace=show_trace,
    )

    for z0 in starting_points
        result = _optimise(objective, z0, optimizer, options)

        push!(runs, result)

        value = Optim.minimum(result)

        if isfinite(value) &&
            value < 1e100 &&
            (isnothing(best) || value < Optim.minimum(best))
            best = result
        end
    end

    theta = unpack(Optim.minimizer(best))
    ll = calibration_loglikelihood(problem, theta)

    isfinite(ll) ||
        error("Optimiser returned invalid likelihood")

    if method == :map
        lp = calibration_logprior(problem, theta)

        return MAPFit(
            problem,
            theta,
            lp,
            ll,
            lp + ll,
            Optim.converged(best),
            best,
            runs,
        )
    else
        return MLEFit(
            problem,
            theta,
            fixed,
            ll,
            Optim.converged(best),
            best,
            runs,
            length(names),
        )
    end
end

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

"""
    fit_map(problem; fixed=NamedTuple(), kwargs...)

Maximise the log posterior in original parameter
coordinates, optionally fixing selected parameters.
"""
function fit_map(
    problem::CalibrationProblem;
    kwargs...,
)
    return _fit_calibration(
        problem;
        method=:map,
        kwargs...,
    )
end

"""
    fit_mle(problem; fixed=NamedTuple(), kwargs...)

Maximise the likelihood. Priors determine parameter
supports and starting points, but do not contribute
to the objective.
"""
function fit_mle(
    problem::CalibrationProblem;
    kwargs...,
)
    return _fit_calibration(
        problem;
        method=:mle,
        kwargs...,
    )
end

# Shared prediction dispatch.
predict(fit::AbstractCalibrationFit) =
    predict(fit.problem, fit.parameters)
