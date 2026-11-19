
"""
Construct the DoddExtended initial state directly.

The initial distribution follows the same heuristic
previously used through the Dodd reference model.

All initial active TB is treatment-history 0.
Recovered individuals occupy the common R compartment.
"""
function initial_state(
    model::DoddExtendedTBModel,
    params::DoddExtendedTBParameters,
    demography::AbstractDemography;
    year::Real,
)
    layout = state_layout(model)
    nages = layout.nages
    stride = nstates(layout)

    population = get_population(demography, year)

    length(population) == nages ||
        throw(DimensionMismatch(
            "Population length does not match model"
        ))

    all(v -> isfinite(v) && v >= 0, population) ||
        throw(ArgumentError("Invalid initial population"))

    total = sum(population)
    total > 0 ||
        throw(ArgumentError("Population must be positive"))

    weights = population ./ total

    ν = sum(weights .* params.nu)
    ω = sum(weights .* params.omega)

    duration = one(params.beta) ./ (
        params.self_cure .+ params.tb_mortality
    )

    mean_duration = sum(weights .* duration)

    Λ0 = params.Lambda0
    detection = detection_probability(
        params.detection, year
    )

    f0 = (ω + ν + params.psi * Λ0) / params.kappa
    d0 = 2 * ν * mean_duration * (1 - detection)

    f0 + 4 * d0 / 3 <= 1 ||
        throw(ArgumentError(
            "Initialisation produces negative latent population"
        ))

    T = promote_type(
        eltype(population),
        typeof(params.beta),
    )

    u0 = zeros(T, stride * nages)

    @inbounds for a in 1:nages
        age = a - 1
        N = population[a]
        base = (a - 1) * stride

        fL = -expm1(-age * Λ0)

        S0 = N * (1 - fL)
        E0 = N * fL * f0 / 2
        L0 = N * fL * (1 - f0 - 4 * d0 / 3)
        I0 = N * fL * d0
        T0 = N * fL * d0 / 3
        R0 = N * fL * f0 / 2

        u0[base+DE.S] = S0
        u0[base+DE.E0] = E0
        u0[base+DE.L0] = L0

        for k in 1:3
            u0[base+DE.I0SP+k-1] =
                params.presentation[a, k] * I0
        end

        u0[base+DE.T] = T0
        u0[base+DE.R] = R0
    end

    return u0
end
