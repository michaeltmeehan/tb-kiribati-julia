using TBKiribatiJulia
using Distributions: Normal, truncated

"""
    kiribati_observation_model(;
        annual_effects=true,
        sigma_sd=0.3,
        tau_sd=0.2,
        quadrature_order=30,
    )

Construct the standard Kiribati notification
observation model.

These are analysis defaults, not model-wide priors.
"""
function kiribati_observation_model(;
    annual_effects::Bool=true,
    sigma_sd::Real=0.3,
    tau_sd::Real=0.2,
    quadrature_order::Int=30,
)
    sigma_sd > 0 && isfinite(sigma_sd) ||
        throw(ArgumentError("Invalid sigma_sd"))

    sigma_prior = truncated(
        Normal(0.0, sigma_sd), 0.0, Inf
    )

    if !annual_effects
        return NB2Observation(sigma_prior)
    end

    tau_sd > 0 && isfinite(tau_sd) ||
        throw(ArgumentError("Invalid tau_sd"))

    tau_prior = truncated(
        Normal(0.0, tau_sd), 0.0, Inf
    )

    return AnnualRandomEffectNB2Observation(
        sigma_prior,
        tau_prior;
        quadrature_order,
    )
end