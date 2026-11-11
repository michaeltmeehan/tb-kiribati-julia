"""
    initial_state(model::DoddTBModel, population, Λ0;
                  kappa, nu, omega, psi, duration, detection)

Construct the heuristic initial condition from Dodd et al.,
Supplementary Methods §1.3.4.

`Λ0` is the initial annual force of infection.
`population` contains the initial population in each single-year age class.

All rates are expressed in inverse years; `duration` is in years.
"""
function initial_state(
    model::DoddTBModel,
    population::AbstractVector,
    Λ0::Real;
    kappa::Real,
    nu::Real,
    omega::Real,
    psi::Real,
    duration::Real,
    detection::Real,
)
    layout = state_layout(model)

    length(population) == layout.nages ||
        throw(DimensionMismatch("Incorrect population length"))

    all(x -> isfinite(x) && x >= 0, population) ||
        throw(ArgumentError("Population must be finite and nonnegative"))

    all(x -> isfinite(x) && x >= 0,
        (Λ0, nu, omega, psi)) ||
        throw(ArgumentError("Rates must be finite and nonnegative"))

    isfinite(kappa) && kappa > 0 ||
        throw(ArgumentError("kappa must be positive"))

    isfinite(duration) && duration > 0 ||
        throw(ArgumentError("duration must be positive"))

    isfinite(detection) && 0 <= detection <= 1 ||
        throw(ArgumentError("detection must lie in [0, 1]"))

    f0 = (omega + nu + psi * Λ0) / kappa
    d0 = 2 * nu * duration * (1 - detection)

    f0 + 4 * d0 / 3 <= 1 ||
        throw(ArgumentError(
            "Initialisation produces a negative latent population"
        ))

    u = zeros(Float64, nstates(layout) * layout.nages)

    @inbounds for a in 1:layout.nages
        age = a - 1
        N = population[a]

        fL = -expm1(-age * Λ0)

        base = (a - 1) * nstates(layout)

        u[base + S] = N * (1 - fL)
        u[base + E] = N * fL * f0 / 2
        u[base + L] = N * fL * (1 - f0 - 4 * d0 / 3)
        u[base + I] = N * fL * d0
        u[base + T] = N * fL * d0 / 3
        u[base + R] = N * fL * f0 / 2
    end

    return u
end


"""
    initial_state(model, population, Λ0, params; year=2023)

Construct the Dodd heuristic initial condition using the
model's epidemiological parameters.

The scalar nu, omega and duration values required by the
original heuristic are approximated by population-weighted
averages of the age-specific parameters.
"""
function initial_state(
    model::DoddTBModel,
    population::AbstractVector,
    Λ0::Real,
    params::DoddTBParameters;
    year::Real,
)
    weights = population ./ sum(population)

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

    return initial_state(
        model,
        population,
        Λ0;
        kappa = params.kappa,
        nu = nu,
        omega = omega,
        psi = params.psi,
        duration = mean_duration,
        detection = detection,
    )
end