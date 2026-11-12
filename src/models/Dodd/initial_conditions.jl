
"""
    initial_state(
        model::DoddTBModel,
        params::DoddTBParameters,
        demography::AbstractDemography;
        year,
    )

Construct the heuristic initial condition from Dodd et al. (2023),
Supplementary Methods §1.3.4.

The initial population is obtained from `demography` at `year`.
All epidemiological parameters, including the initial force of
infection (Lambda0), are obtained from `params`.

Age-specific parameters are reduced to population-weighted averages
where the original heuristic requires scalar quantities.

The late-latent initial condition includes a correction to the
published expression to ensure population conservation.
"""
function initial_state(
    model::DoddTBModel,
    params::DoddTBParameters,
    demography::AbstractDemography;
    year::Real,
)
    layout = state_layout(model)

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = get_population(demography, year)

    length(population) == layout.nages ||
        throw(DimensionMismatch(
            "Population length does not match model age classes"
        ))

    all(x -> isfinite(x) && x >= 0, population) ||
        throw(ArgumentError(
            "Population must be finite and nonnegative"
        ))

    total_population = sum(population)

    total_population > 0 ||
        throw(ArgumentError(
            "Initial population must be positive"
        ))

    # --------------------------------------------------------
    # Initialisation parameters
    # --------------------------------------------------------

    Λ0 = params.Lambda0

    isfinite(Λ0) && Λ0 >= 0 ||
        throw(ArgumentError(
            "Lambda0 must be finite and nonnegative"
        ))

    weights = population ./ total_population

    nu = sum(weights .* params.nu)
    omega = sum(weights .* params.omega)

    duration = 1.0 ./ (
        params.self_cure .+ params.tb_mortality
    )

    mean_duration = sum(weights .* duration)

    detection = detection_probability(
        params.detection,
        year,
    )

    # --------------------------------------------------------
    # Dodd initialisation heuristic
    # --------------------------------------------------------

    f0 = (
        omega + nu + params.psi * Λ0
    ) / params.kappa

    d0 = 2 * nu * mean_duration * (1 - detection)

    f0 + 4 * d0 / 3 <= 1 ||
        throw(ArgumentError(
            "Initialisation produces a negative latent population"
        ))

    # --------------------------------------------------------
    # Construct initial state
    # --------------------------------------------------------

    u0 = zeros(Float64, nstates(layout) * layout.nages)

    @inbounds for a in 1:layout.nages

        age = a - 1
        N = population[a]

        fL = -expm1(-age * Λ0)

        base = (a - 1) * nstates(layout)

        u0[base+S] = N * (1 - fL)
        u0[base+E] = N * fL * f0 / 2
        u0[base+L] = N * fL * (1 - f0 - 4 * d0 / 3)
        u0[base+I] = N * fL * d0
        u0[base+T] = N * fL * d0 / 3
        u0[base+R] = N * fL * f0 / 2

    end

    return u0
end
