
using Distributions: NegativeBinomial, logpdf
using LinearAlgebra: SymTridiagonal, eigen

"""
Independent NB2 observations:

    Var(Y) = mu + sigma_obs^2 * mu^2
"""
struct NB2Observation{D}
    sigma_prior::D
end


"""
    AnnualRandomEffectNB2Observation(
        sigma_prior,
        tau_prior;
        quadrature_order=30,
    )

NB2 observation model with a shared annual lognormal effect.

    Z_t ~ Normal(0, 1)

    m_t = exp(tau_year * Z_t - tau_year^2 / 2)

    Y[t,g] | Z_t ~ NB2(mu[t,g] * m_t, sigma_obs)

The random effect is integrated out by Gaussian quadrature.
The multiplier is centred so its expectation is one.
"""
struct AnnualRandomEffectNB2Observation{D,E}
    sigma_prior::D
    tau_prior::E
    nodes::Vector{Float64}
    weights::Vector{Float64}
end


"""
    calibration_normal_quadrature(n)

Nodes and weights for integration against N(0,1).
Weights sum to one.
"""
function calibration_normal_quadrature(n::Integer)
    n >= 1 || throw(ArgumentError(
        "quadrature_order must be positive"
    ))

    offdiagonal = sqrt.(Float64.(1:(n-1)))

    eig = eigen(
        SymTridiagonal(
            zeros(Float64, n),
            offdiagonal,
        )
    )

    nodes = collect(eig.values)
    weights = collect(abs2.(eig.vectors[1, :]))

    return nodes, weights
end


function AnnualRandomEffectNB2Observation(
    sigma_prior,
    tau_prior;
    quadrature_order::Int=30,
)
    nodes, weights =
        calibration_normal_quadrature(quadrature_order)

    return AnnualRandomEffectNB2Observation(
        sigma_prior,
        tau_prior,
        nodes,
        weights,
    )
end


"""
    nb2_loglikelihood(y, mu, sigma)

NB2 log probability for a count y, with

    Var(Y) = mu + sigma^2 * mu^2
"""
function nb2_loglikelihood(
    y::Integer,
    mu::Real,
    sigma::Real,
)
    if y < 0 ||
        !isfinite(mu) || mu < 0 ||
        !isfinite(sigma) || sigma <= 0
        return -Inf
    end

    if mu == 0
        return y == 0 ? 0.0 : -Inf
    end

    phi = inv(sigma^2)

    if !isfinite(phi) || phi <= 0
        return -Inf
    end

    p = phi / (phi + mu)

    return logpdf(
        NegativeBinomial(phi, p),
        y,
    )
end


# ------------------------------------------------------------
# Observation-model priors
# ------------------------------------------------------------

observation_priors(obs::NB2Observation) = (
    sigma_obs=obs.sigma_prior,
)

observation_priors(
    obs::AnnualRandomEffectNB2Observation
) = (
    sigma_obs=obs.sigma_prior,
    tau_year=obs.tau_prior,
)


# ------------------------------------------------------------
# Marginal observation variances
# ------------------------------------------------------------

function observation_variance(
    ::NB2Observation,
    mu::Real,
    theta,
)
    return mu + theta.sigma_obs^2 * mu^2
end


function observation_variance(
    ::AnnualRandomEffectNB2Observation,
    mu::Real,
    theta,
)
    sigma = theta.sigma_obs
    tau = theta.tau_year

    return mu + mu^2 * (
        (1 + sigma^2) * exp(tau^2) - 1
    )
end


# ------------------------------------------------------------
# Independent NB2 likelihood for one year
# ------------------------------------------------------------

"""
    observation_year_loglikelihood(obs, rows, theta)

Joint log-likelihood for all reported age categories
within one year.
"""
function observation_year_loglikelihood(
    ::NB2Observation,
    rows,
    theta,
)
    ll = 0.0

    for row in rows
        term = nb2_loglikelihood(
            row.observed,
            row.expected,
            theta.sigma_obs,
        )

        isfinite(term) || return -Inf

        ll += term
    end

    return ll
end


# ------------------------------------------------------------
# Shared annual random-effect likelihood
# ------------------------------------------------------------

function observation_year_loglikelihood(
    obs::AnnualRandomEffectNB2Observation,
    rows,
    theta,
)
    sigma = theta.sigma_obs
    tau = theta.tau_year

    if !isfinite(sigma) || sigma <= 0 ||
        !isfinite(tau) || tau < 0
        return -Inf
    end

    # Exact reduction to independent NB2 when tau = 0.
    if tau == 0
        return observation_year_loglikelihood(
            NB2Observation(obs.sigma_prior),
            rows,
            theta,
        )
    end

    logs = Vector{Float64}(
        undef,
        length(obs.nodes),
    )

    for k in eachindex(obs.nodes)

        multiplier = exp(
            tau * obs.nodes[k] - tau^2 / 2
        )

        ll = log(obs.weights[k])

        for row in rows
            term = nb2_loglikelihood(
                row.observed,
                row.expected * multiplier,
                sigma,
            )

            isfinite(term) || return -Inf

            ll += term
        end

        logs[k] = ll
    end

    # Numerically stable log-sum-exp.
    m = maximum(logs)

    isfinite(m) || return m

    return m + log(
        sum(exp(x - m) for x in logs)
    )
end
