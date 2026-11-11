
using Distributions

"""
    dodd_priors()

Return the epidemiological parameter priors reported by
Dodd et al. (2023), Supplementary Table 1.

Distribution parameters follow Distributions.jl conventions.
"""
function priors(model::DoddTBModel)
    return (
        # Transmission
        beta = LogNormal(log(10), 0.75),
        psi = Beta(20.7, 77.9),
        Lambda0 = LogNormal(log(3e-2), 0.75),

        # Progression
        kappa = LogNormal(0.62, 0.068),
        epsilon = LogNormal(-2.837, 0.32),
        epsilon_child = Beta(5.153, 21.967),
        nu = LogNormal(-6.89, 0.58),
        omega = LogNormal(-3.95, 0.27),

        # Detection
        detection_trend = Gamma(0.5, 0.04),
        detection_or_0_4 = LogNormal(-0.100, 0.626),
        detection_or_5_14 = LogNormal(-0.567, 0.458),

        # Untreated disease
        duration = LogNormal(1.1, 0.2),
        untreated_cfr = Beta(25.48, 33.78),

        # Treatment
        treatment_cfr = Beta(2.71, 87.55),
    )
end
