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


function model_incidence_by_age_group(
    beta,
    progression_65_plus;
    infectiousness_weights = (0.2, 0.5, 0.4, 1.0),
)

    # Construct parameters
    params = make_parameters(
        CONTACT;
        beta = beta,
        progression_65_plus = progression_65_plus,
    )

    params.infectiousness_weights = infectiousness_weights

    # Initial conditions
    population = get_population(STATIC_YEAR)
    u0 = TBKiribatiJulia.initial_state(population)

    # Simulation
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

    # Raw incidence by single-year age
    _, incidence_by_age = raw_annual_incidence_by_age(sol)

    # Aggregate into observed age groups
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


function calibration_loss(beta, progression_65_plus)

    predicted = model_incidence_by_age_group(
        beta,
        progression_65_plus,
    )

    residuals =
        log.(predicted) .-
        log.(CALIBRATION_INCIDENCE)

    return sum(abs2, residuals)
end

predicted = model_incidence_by_age_group(
    0.75,
    2.4,
)

calibration_loss(
    0.75,
    2.4,
)

using Optim

# Objective function for Optim.
# Work on the log scale so beta and progression_65_plus are constrained > 0.
function objective(x)

    beta = exp(x[1])
    progression_65_plus = exp(x[2])

    return calibration_loss(
        beta,
        progression_65_plus,
    )
end


# Initial guesses
beta0 = 0.75
progression_65_plus0 = 2.4

x0 = log.([
    beta0,
    progression_65_plus0,
])


# Run optimization
result = optimize(
    objective,
    x0,
    NelderMead(),
    Optim.Options(
        iterations = 500,
        show_trace = true,
        show_every = 10,
    ),
)


# Extract fitted parameters
xhat = Optim.minimizer(result)

beta_hat = exp(xhat[1])
progression_65_plus_hat = exp(xhat[2])

println("Converged: ", Optim.converged(result))
println("Minimum loss: ", Optim.minimum(result))
println("beta = ", beta_hat)
println("progression_65_plus = ", progression_65_plus_hat)


# Predicted age-specific incidence at optimum
fitted_incidence = model_incidence_by_age_group(
    beta_hat,
    progression_65_plus_hat,
)

println()
println("Observed incidence:  ", CALIBRATION_INCIDENCE)
println("Predicted incidence: ", fitted_incidence)