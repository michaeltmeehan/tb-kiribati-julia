using CSV
using DataFrames
using Optim
using Distributions
using Statistics


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))

const NOTIFICATION_FILE =
    joinpath(DATA_DIR, "tb_notifications_by_age.csv")

const INCIDENCE_FILE =
    joinpath(DATA_DIR, "tb_incidence.csv")

const NOTIFICATIONS = CSV.read(
    NOTIFICATION_FILE,
    DataFrame,
)

const INCIDENCE = CSV.read(
    INCIDENCE_FILE,
    DataFrame,
)

# Model age groups used in the calibration likelihood.
const CALIBRATION_AGE_BREAKS = [0, 5, 15, 25, 35, 45, 55, 65, 96]

const CALIBRATION_AGE_LABELS = [
    "0-4",
    "5-14",
    "15-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]

# Children are assumed to be detected at one fifth the adult rate.
# Only this relative detection rate is needed for the age-composition
# likelihood; the absolute child/adult rates are derived afterwards from the
# WHO overall notification/incidence ratio.
const INFANT_TO_ADULT_DETECTION_RATIO = 1.0

const CHILD_TO_ADULT_DETECTION_RATIO = 1.0

const RELATIVE_DETECTION = [
    INFANT_TO_ADULT_DETECTION_RATIO,   # 0-4
    CHILD_TO_ADULT_DETECTION_RATIO,   # 5-14
    1.0,                              # 15-24
    1.0,                              # 25-34
    1.0,                              # 35-44
    1.0,                              # 45-54
    1.0,                              # 55-64
    1.0,                              # 65+
]

# Notification data are available from 2013 onwards. Restrict the incidence
# likelihood to the same period so that both components describe the same
# calibration era.
const CALIBRATION_YEARS = sort(intersect(
    unique(Int.(NOTIFICATIONS.year)),
    unique(Int.(INCIDENCE.year)),
))


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------

# The current calibration model uses equilibrium demography and constant
# epidemiological parameters. The annual WHO observations are therefore
# treated as repeated observations of a common equilibrium process rather than
# as targets for year-to-year model fluctuations.

tinit = 1800.0
tfinal = 2024.0
times = tinit:1.0:tfinal


# ---------------------------------------------------------------------------
# Model-predicted true incidence
# ---------------------------------------------------------------------------

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


function model_total_incidence_rate(predicted_incidence)

    population = get_population(EQUILIBRIUM_YEAR)

    return 1e5 * sum(predicted_incidence) / sum(population)
end


# ---------------------------------------------------------------------------
# Notification-data helpers
# ---------------------------------------------------------------------------

function case_count(year_data, age_group)

    values = year_data[
        year_data.age_group .== age_group,
        :cases,
    ]

    isempty(values) && return missing

    return only(values)
end


function notification_observation(year, predicted_incidence)

    year_data = NOTIFICATIONS[
        NOTIFICATIONS.year .== year,
        :,
    ]

    isempty(year_data) && error("No notification data for year $year")

    counts = Int[]
    expected_weights = Float64[]
    labels = String[]

    # Prefer the separate 0-4 and 5-14 observations whenever both are
    # available. Fall back to the 0-14 aggregate otherwise. Never use both,
    # because that would double-count the same childhood notifications.
    y04 = case_count(year_data, "0-4")
    y514 = case_count(year_data, "5-14")
    y014 = case_count(year_data, "0-14")

    if !ismissing(y04) && !ismissing(y514)

        push!(counts, Int(y04))
        push!(expected_weights,
            INFANT_TO_ADULT_DETECTION_RATIO * predicted_incidence[1])
        push!(labels, "0-4")

        push!(counts, Int(y514))
        push!(expected_weights,
            CHILD_TO_ADULT_DETECTION_RATIO * predicted_incidence[2])
        push!(labels, "5-14")

    elseif !ismissing(y014)

        push!(counts, Int(y014))
        push!(expected_weights,
            INFANT_TO_ADULT_DETECTION_RATIO * predicted_incidence[1] +
            CHILD_TO_ADULT_DETECTION_RATIO * predicted_incidence[2])
        push!(labels, "0-14")

    else
        error("No usable childhood notification data for year $year")
    end

    # Adult age groups.
    for g in 3:length(CALIBRATION_AGE_LABELS)

        age_group = CALIBRATION_AGE_LABELS[g]
        y = case_count(year_data, age_group)

        if !ismissing(y)
            push!(counts, Int(y))
            push!(expected_weights,
                RELATIVE_DETECTION[g] * predicted_incidence[g])
            push!(labels, age_group)
        end
    end

    return counts, expected_weights, labels
end


# ---------------------------------------------------------------------------
# Likelihood components
# ---------------------------------------------------------------------------

function notification_loglikelihood(predicted_incidence)

    loglik = 0.0

    for year in CALIBRATION_YEARS

        counts, weights, _ =
            notification_observation(year, predicted_incidence)

        if any(x -> !isfinite(x) || x <= 0, weights)
            return -Inf
        end

        probabilities = weights ./ sum(weights)
        total_cases = sum(counts)

        loglik += logpdf(
            Multinomial(total_cases, probabilities),
            counts,
        )
    end

    return loglik
end


function incidence_loglikelihood(predicted_incidence)

    predicted_rate = model_total_incidence_rate(predicted_incidence)

    if !isfinite(predicted_rate) || predicted_rate <= 0
        return -Inf
    end

    loglik = 0.0

    for row in eachrow(INCIDENCE)

        year = Int(row.year)
        year in CALIBRATION_YEARS || continue

        observed = Float64(row.incidence_rate)
        lower = Float64(row.incidence_lower)
        upper = Float64(row.incidence_upper)

        if observed <= 0 || lower <= 0 || upper <= 0
            continue
        end

        # Approximate the WHO uncertainty interval by a log-normal sampling
        # distribution. The central WHO estimate is treated as the median.
        sigma =
            (log(upper) - log(lower)) /
            (2 * 1.96)

        loglik += logpdf(
            LogNormal(log(predicted_rate), sigma),
            observed,
        )
    end

    return loglik
end


function calibration_loss(
    beta,
    progression_child,
    progression_5_14,
    progression_15_64,
    progression_65_plus,
)

    predicted_incidence = model_incidence_by_age_group(
        beta,
        progression_child,
        progression_5_14,
        progression_15_64,
        progression_65_plus,
    )

    if any(x -> !isfinite(x) || x <= 0, predicted_incidence)
        return Inf
    end

    notification_ll =
        notification_loglikelihood(predicted_incidence)

    incidence_ll =
        incidence_loglikelihood(predicted_incidence)

    if !isfinite(notification_ll) || !isfinite(incidence_ll)
        return Inf
    end

    return -(notification_ll + incidence_ll)
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


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

x0 = [
    0.75,   # beta
    2.4,    # progression_child
    0.05,   # progression_5_14
    0.1,    # progression_15_64
    2.4,    # progression_65_plus
]


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

lower = [
    0.01,   # beta
    0.01,   # progression_child
    0.01,   # progression_5_14
    0.01,   # progression_15_64
    0.01,   # progression_65_plus
]

upper = [
     5.0,   # beta
    10.0,   # progression_child
    10.0,   # progression_5_14
    10.0,   # progression_15_64
    10.0,   # progression_65_plus
]


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

xhat = Optim.minimizer(result)

beta_hat = xhat[1]
progression_child_hat = xhat[2]
progression_5_14_hat = xhat[3]
progression_15_64_hat = xhat[4]
progression_65_plus_hat = xhat[5]

println()
println("Converged: ", Optim.converged(result))
println("Minimum negative log-likelihood: ", Optim.minimum(result))
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

fitted_total_rate =
    model_total_incidence_rate(fitted_incidence)

println()
println(
    "Fitted equilibrium total incidence rate = ",
    round(fitted_total_rate, digits = 2),
    " per 100,000",
)

println()
println("Fitted true incidence by age group:")

for g in eachindex(CALIBRATION_AGE_LABELS)
    println(
        CALIBRATION_AGE_LABELS[g],
        ": ",
        round(fitted_incidence[g], digits = 2),
    )
end


# ---------------------------------------------------------------------------
# Implied absolute detection probabilities
# ---------------------------------------------------------------------------

infant_incidence = fitted_incidence[1]
child_incidence = fitted_incidence[2]
adult_incidence = sum(fitted_incidence[3:end])

total_incidence =
    infant_incidence +
    child_incidence +
    adult_incidence

implied_detection = DataFrame(
    year = Int[],
    overall = Float64[],
    infant = Float64[],
    child = Float64[],
    adult = Float64[],
)

for row in eachrow(INCIDENCE)

    year = Int(row.year)
    year in CALIBRATION_YEARS || continue

    q_overall = Float64(row.detection_fraction)

    q_adult =
        q_overall * total_incidence /
        (
            INFANT_TO_ADULT_DETECTION_RATIO * infant_incidence +
            CHILD_TO_ADULT_DETECTION_RATIO * child_incidence +
            adult_incidence
        )

    q_infant =
        INFANT_TO_ADULT_DETECTION_RATIO * q_adult

    q_child =
        CHILD_TO_ADULT_DETECTION_RATIO * q_adult

    push!(
        implied_detection,
        (
            year = year,
            overall = q_overall,
            infant = q_infant,
            child = q_child,
            adult = q_adult,
        ),
    )
end

println()
println("Implied detection probabilities:")

for row in eachrow(implied_detection)
    println(
        row.year,
        ": overall = ", round(row.overall, digits = 3),
        ", 0-4 = ", round(row.infant, digits = 3),
        ", 5-14 = ", round(row.child, digits = 3),
        ", >=15 = ", round(row.adult, digits = 3),
    )
end

println()
println(
    "Mean implied detection: 0-4 = ",
    round(mean(implied_detection.infant), digits = 3),
    ", 5-14 = ",
    round(mean(implied_detection.child), digits = 3),
    ", >=15 = ",
    round(mean(implied_detection.adult), digits = 3),
)


# ---------------------------------------------------------------------------
# Age-composition diagnostics
# ---------------------------------------------------------------------------

println()
println("Notification age-composition fit:")

for year in CALIBRATION_YEARS

    counts, weights, labels =
        notification_observation(year, fitted_incidence)

    probabilities = weights ./ sum(weights)
    expected = sum(counts) .* probabilities

    println()
    println(year)

    for i in eachindex(labels)
        println(
            "  ", labels[i],
            ": observed = ", counts[i],
            ", expected = ", round(expected[i], digits = 1),
        )
    end
end
