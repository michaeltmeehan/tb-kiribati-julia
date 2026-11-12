
using Random
using Distributions

"""
    sample_prior(rng::AbstractRNG, model::AbstractTBModel)

Draw a parameter set independently from the distributions
returned by `priors(model)`.

Returns a NamedTuple with the same keys as the prior
specification.

Parameters are sampled on their specified scales;
model-specific transformations are handled separately
by `make_parameters(model, draw)`.
"""
function sample_prior(
    rng::AbstractRNG,
    model::AbstractTBModel,
)
    return map(d -> rand(rng, d), priors(model))
end

"""
    sample_prior(model::AbstractTBModel)

Draw a parameter set using the default random number generator.
"""
sample_prior(model::AbstractTBModel) =
    sample_prior(Random.default_rng(), model)
