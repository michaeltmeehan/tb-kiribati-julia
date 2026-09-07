using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
import DiffEqCallbacks as CB
using Optim


const CALIBRATION_AGE_BREAKS = [0, 15, 25, 35, 45, 55, 65, 96]

const CALIBRATION_INCIDENCE = [
    26.0,
    173.0,
    117.0,
    158.0,
    178.0,
    210.0,
    147.0
]


tinit = 1800.0
tfinal = 2100.0
times = tinit:1.0:tfinal


function annual_update!(integrator)

    if integrator.t <= 2100.0
        _apply_static_demography!(integrator.u)
    else
        _apply_demography!(integrator.u, integrator.t - 1.0)
    end

    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE

        for c in (NEPI + 1):NSTATE
            integrator.u[base + c] = 0.0
        end
    end
end


cb = CB.PresetTimeCallback(times, annual_update!)


function model_incidence_by_age_group(
    beta,
    progression_child,
    progression_5_14,
    progression_15_64,
    progression_65_plus;
    infectiousness_weights = (0.2, 0.5, 0.4, 1.0),
)

    params = make_parameters(
        CONTACT;
        # ones(NAGE, NAGE);
        beta = beta,
        progression_child = progression_child,
        progression_5_14 = progression_5_14,
        progression_15_64 = progression_15_64,
        progression_65_plus = progression_65_plus,
    )

    params.infectiousness_weights = infectiousness_weights

    population = get_population(STATIC_YEAR)
    u0 = TBKiribatiJulia.initial_state(population)

    prob = ODEProblem(
        tb_rhs!,
        u0,
        (tinit, tfinal),
        params,
    )

    sol = solve(
        prob,
        Vern7(),
        callback = cb;
        saveat = times,
    )

    _, incidence_by_age = raw_annual_incidence_by_age(sol)

    predicted = Vector{Float64}(
        undef,
        length(CALIBRATION_AGE_BREAKS) - 1,
    )

    for g in eachindex(predicted)

        lower = CALIBRATION_AGE_BREAKS[g]
        upper = CALIBRATION_AGE_BREAKS[g + 1]

        cases = 0.0
        pop = 0.0

        for age in lower:(upper - 1)
            cases += incidence_by_age[end, age + 1]
            pop += population[age + 1]
        end

        predicted[g] = 1e5 * cases / pop
    end

    return predicted
end


function calibration_loss(
    beta,
    progression_child,
    progression_5_14,
    progression_15_64,
    progression_65_plus,
)

    predicted = model_incidence_by_age_group(
        beta,
        progression_child,
        progression_5_14,
        progression_15_64,
        progression_65_plus,
    )

    if any(x -> !isfinite(x) || x <= 0, predicted)
        return Inf
    end

    residuals =
        log.(predicted) .-
        log.(CALIBRATION_INCIDENCE)

    return sum(abs2, residuals)
end


function objective(x)

    return calibration_loss(
        x[1],   # beta
        x[2],   # progression_child
        x[3],   # progression_5_14
        x[4],   # progression_15_64
        x[5],   # progression_65_plus
    )
end


# Starting values
x0 = [
    0.75,   # beta
    2.4,    # progression_child
    2.0,    # progression_5_14
    0.1,    # progression_15_64
    2.4,    # progression_65_plus
]


# Broad but finite bounds
lower = [
    0.01,   # beta
    0.01,   # progression_child
    0.01,   # progression_5_14
    0.01,   # progression_15_64
    0.01,   # progression_65_plus
]

upper = [
    5.0,    # beta
    10.0,   # progression_child
    10.0,   # progression_5_14
    10.0,   # progression_15_64
    10.0,   # progression_65_plus
]


result = optimize(
    objective,
    lower,
    upper,
    x0,
    Fminbox(NelderMead()),
    Optim.Options(
        iterations = 500,
        show_trace = true,
        show_every = 20,
    ),
)


xhat = Optim.minimizer(result)

beta_hat = xhat[1]
progression_child_hat = xhat[2]
progression_5_14_hat = xhat[3]
progression_15_64_hat = xhat[4]
progression_65_plus_hat = xhat[5]


println()
println("Converged: ", Optim.converged(result))
println("Minimum loss: ", Optim.minimum(result))
println()

println("Fitted parameters:")
println("beta                  = ", beta_hat)
println("progression_child     = ", progression_child_hat)
println("progression_5_14      = ", progression_5_14_hat)
println("progression_15_64     = ", progression_15_64_hat)
println("progression_65_plus   = ", progression_65_plus_hat)


fitted_incidence = model_incidence_by_age_group(
    beta_hat,
    progression_child_hat,
    progression_5_14_hat,
    progression_15_64_hat,
    progression_65_plus_hat,
)


println()
println("Age-group fit:")

for g in eachindex(CALIBRATION_INCIDENCE)

    lower_age = CALIBRATION_AGE_BREAKS[g]
    upper_age = CALIBRATION_AGE_BREAKS[g + 1] - 1

    println(
        lower_age, "-", upper_age,
        ": observed = ",
        round(CALIBRATION_INCIDENCE[g], digits = 2),
        ", predicted = ",
        round(fitted_incidence[g], digits = 2),
    )
end