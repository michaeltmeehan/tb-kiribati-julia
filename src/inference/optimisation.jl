using Random
using Distributions: Normal, cdf, quantile
using Optim

struct MAPFit{P,T,R}
    problem::P
    parameters::T
    log_prior::Float64
    log_likelihood::Float64
    log_posterior::Float64
    converged::Bool
    optim_result::R
    runs::Vector{Any}
end

"""
    fit_map(problem; rng, nstarts=3, iterations=2000)

Multi-start MAP in normal-quantile coordinates. The objective is the posterior
DENSITY IN ORIGINAL PARAMETERS; the coordinate transform introduces no Jacobian.
"""
function fit_map(problem::CalibrationProblem; rng=Random.default_rng(),
    nstarts::Int=3, iterations::Int=2000, show_trace::Bool=false)
    nstarts > 0 && iterations > 0 || throw(ArgumentError("Invalid optimisation settings"))
    names = keys(problem.prior)
    ds = values(problem.prior)
    stdnormal = Normal()
    function unpack(z)
        theta = ntuple(j -> quantile(ds[j], clamp(cdf(stdnormal, z[j]), 1e-10, 1-1e-10)), length(ds))
        return NamedTuple{names}(theta)
    end
    objective(z) = begin
        lp = calibration_logposterior(problem, unpack(z))
        isfinite(lp) ? -lp : 1e100
    end
    runs = Any[]
    best = nothing
    for k in 1:nstarts
        z0 = k == 1 ? zeros(length(ds)) : randn(rng, length(ds))
        result = Optim.optimize(objective, z0, Optim.NelderMead(),
            Optim.Options(iterations=iterations, show_trace=show_trace))
        push!(runs, result)
        if isfinite(Optim.minimum(result)) &&
            (isnothing(best) || Optim.minimum(result) < Optim.minimum(best))
            best = result
        end
    end
    isnothing(best) && error("No finite MAP optimisation result")
    theta = unpack(Optim.minimizer(best))
    lp = calibration_logprior(problem, theta)
    ll = calibration_loglikelihood(problem, theta)
    isfinite(lp+ll) || error("MAP optimisation did not find a finite posterior")
    return MAPFit(problem, theta, lp, ll, lp+ll, Optim.converged(best), best, runs)
end

"""Predict annual incidence, notification and mortality trajectories at a MAP fit."""
predict(fit::MAPFit) = predict(fit.problem, fit.parameters)
