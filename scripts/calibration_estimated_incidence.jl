using Optim

# Estimated number of TB cases by age and sex, 2024
# > men
#  0-14 15-24 25-34 35-44 45-54 55-64   65+ 
#   220   120   100    72    72    68    38 
# > women
#  0-14 15-24 25-34 35-44 45-54 55-64   65+ 
#   190   100    83    58    47    53    43 
# > men + women
#  0-14 15-24 25-34 35-44 45-54 55-64   65+ 
#   410   220   183   130   119   121    81 

const CALIBRATION_AGE_BREAKS = [0, 15, 25, 35, 45, 55, 65, 96]

# Data from Hill et al. 10.5588/ijtid.14.0007
const CALIBRATION_INCIDENCE = [
    26.0,   # 0-14
    173.0,  # 15-24
    117.0,  # 25-34
    158.0,  # 35-44
    178.0,  # 45-54
    210.0,  # 55-64
    147.0   # 65+
]

# Data from WHO, 2024
const CALIBRATION_CASES = [
    410.0,  # 0-14
    220.0,  # 15-24
    183.0,  # 25-34
    130.0,  # 35-44
    119.0,  # 45-54
    121.0,  # 55-64
    81.0,   # 65+
]

# Further disaggregated data from WHO 2024
# const CALIBRATION_CASES = [
#     141.0,    # 0-4
#     134.0,    # 5-9
#     136.0,    # 10-14
#     220.0,    # 15-24
#     183.0,    # 25-34
#     130.0,    # 35-44
#     119.0,    # 45-54
#     121.0,    # 55-64
#     81.0,     # 65+
# ]

const CALIBRATION_LOWER = [
    # lower uncertainty limits
]

const CALIBRATION_UPPER = [
    # upper uncertainty limits
]

const CALIBRATION_SIGMA =
    (log.(CALIBRATION_UPPER) .- log.(CALIBRATION_LOWER)) ./ (2 * 1.96)


tinit = 1800.0
tfinal = 2024.0
times = tinit:1.0:tfinal



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
        infectiousness_weights = infectiousness_weights,
    )

    population = get_population(EQUILIBRIUM_YEAR)
    u0 = seeded_initial_state(population)

    sol = simulate(
    params;
    tspan = (times[1], times[end]),
    u0 = u0,
    saveat = times,
    demography = :equilibrium,
)

    _, incidence_by_age = raw_annual_incidence_by_age(sol)

return aggregate_age_groups(
    incidence_by_age[end, :],
    CALIBRATION_AGE_BREAKS,
)
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
        # log.(CALIBRATION_INCIDENCE)
        log.(CALIBRATION_CASES)

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
    # 2.0,    # progression_5_14
    0.05,    # progression_5_14
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
    # 10.0,   # progression_5_14
    0.2,   # progression_5_14   -> Imposed upper bound to restrict infection rate in the 5-14 cohort
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

# for g in eachindex(CALIBRATION_INCIDENCE)
for g in eachindex(CALIBRATION_CASES)

    lower_age = CALIBRATION_AGE_BREAKS[g]
    upper_age = CALIBRATION_AGE_BREAKS[g + 1] - 1

    println(
        lower_age, "-", upper_age,
        ": observed = ",
        # round(CALIBRATION_INCIDENCE[g], digits = 2),
        round(CALIBRATION_CASES[g], digits = 2),
        ", predicted = ",
        round(fitted_incidence[g], digits = 2),
    )
end