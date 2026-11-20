using TBKiribatiJulia
using Distributions
using Optim
using LinearAlgebra

include("calibration_data.jl")


# ---------------------------------------------------------------------------
# Parameters included in this calibration
# ---------------------------------------------------------------------------

const MODEL_PARAMETERS = (
    :beta,
    :progression_child,
    :progression_5_14,
    :progression_15_64,
    :progression_65_plus,
)

const N_MODEL_PARAMETERS = length(MODEL_PARAMETERS)


# ---------------------------------------------------------------------------
# Model and likelihood wrappers
# ---------------------------------------------------------------------------

function model_parameter_namedtuple(x)

    length(x) >= N_MODEL_PARAMETERS ||
        error("Parameter vector has incorrect length")

    return NamedTuple{MODEL_PARAMETERS}(
        Tuple(x[1:N_MODEL_PARAMETERS]),
    )
end


function unpack_parameters(x)

    length(x) == N_MODEL_PARAMETERS + 2 ||
        error("Parameter vector has incorrect length")

    model_parameters = model_parameter_namedtuple(x)

    sigma_obs = x[N_MODEL_PARAMETERS + 1]
    tau_year = x[N_MODEL_PARAMETERS + 2]

    return model_parameters, sigma_obs, tau_year
end



function model_predictions(model_parameters)

    return model_incidence_by_age_group(
        model_parameters;
        age_breaks = CALIBRATION_AGE_BREAKS,
        fixed_parameters = FIXED_PARAMETERS,
        tinit = TINIT,
        tfinal = TFINAL,
    )
end



function loglikelihood(model_parameters, sigma_obs, tau_year)

    return calibration_loglikelihood(
        model_parameters;
        sigma_obs = sigma_obs,
        tau_year = tau_year,
        quadrature_order = 30,
        notification_observations = NOTIFICATION_OBSERVATIONS,
        relative_detection = RELATIVE_DETECTION,
        age_breaks = CALIBRATION_AGE_BREAKS,
        fixed_parameters = FIXED_PARAMETERS,
        tinit = TINIT,
        tfinal = TFINAL,
    )
end


function objective(x)

    model_parameters, sigma_obs, tau_year =
        unpack_parameters(x)

    ll = loglikelihood(
        model_parameters,
        sigma_obs,
        tau_year,
    )

    return isfinite(ll) ? -ll : Inf
end



# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

# Transmission starting values are close to the centres of the current priors.
# sigma_obs = 0.1 corresponds to phi = 100.

x0 = [
    1.0,    # beta
    1.5,    # progression_child
    0.1,    # progression_5_14
    0.2,    # progression_15_64
    0.4,    # progression_65_plus
    0.1,    # sigma_obs
    0.2,    # tau_year
]


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

lower = [
    0.01,    # beta
    0.01,    # progression_child
    0.01,    # progression_5_14
    0.01,    # progression_15_64
    0.01,    # progression_65_plus
    1.0e-4,  # sigma_obs; small positive value approximates Poisson limit
    0.0,     # tau_year
]

upper = [
     5.0,    # beta
    10.0,    # progression_child
    10.0,    # progression_5_14
    10.0,    # progression_15_64
    10.0,    # progression_65_plus
     2.0,    # sigma_obs
     1.5,    # tau_year
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
        iterations = 1000,
        show_trace = true,
        show_every = 20,
    ),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

xhat = Optim.minimizer(result)
fitted_parameters, fitted_sigma_obs, fitted_tau_year =
    unpack_parameters(xhat)
fitted_phi = inv(fitted_sigma_obs^2)

println()
println("Converged: ", Optim.converged(result))
println("Minimum negative log-likelihood: ", Optim.minimum(result))
println()
println("Fitted parameters:")

for (parameter, value) in pairs(fitted_parameters)
    println(
        rpad(string(parameter), 25),
        " = ",
        value,
    )
end

println(
    rpad("sigma_obs", 25),
    " = ",
    fitted_sigma_obs,
)

println(
    rpad("phi", 25),
    " = ",
    fitted_phi,
)

println(
    rpad("tau_year", 25),
    " = ",
    fitted_tau_year,
)

println(
    rpad("Annual multiplier CV", 25),
    " = ",
    sqrt(expm1(fitted_tau_year^2)),
)



# ---------------------------------------------------------------------------
# Original versus extended likelihood comparison
# ---------------------------------------------------------------------------

# Previous MLE from the independent negative-binomial model.
original_parameters = (
    beta = 0.5340911859575941,
    progression_child = 1.455908702065598,
    progression_5_14 = 0.6027272527421328,
    progression_15_64 = 0.2584611903861628,
    progression_65_plus = 0.38331796427454745,
)

original_sigma_obs = 0.3351982721420657

ll_original = loglikelihood(
    original_parameters,
    original_sigma_obs,
    0.0,
)

ll_extended = -Optim.minimum(result)

k_original = N_MODEL_PARAMETERS + 1
k_extended = N_MODEL_PARAMETERS + 2

aic_original = -2 * ll_original + 2 * k_original
aic_extended = -2 * ll_extended + 2 * k_extended

println()
println("Likelihood comparison")
println(
    "  Original log-likelihood = ",
    round(ll_original, digits = 3),
)
println(
    "  Extended log-likelihood = ",
    round(ll_extended, digits = 3),
)
println(
    "  Original AIC = ",
    round(aic_original, digits = 3),
)
println(
    "  Extended AIC = ",
    round(aic_extended, digits = 3),
)
println(
    "  Delta AIC (extended - original) = ",
    round(aic_extended - aic_original, digits = 3),
)
println(
    "  Likelihood-ratio statistic = ",
    round(2 * (ll_extended - ll_original), digits = 3),
)


# ---------------------------------------------------------------------------
# Fitted incidence
# ---------------------------------------------------------------------------

fitted_incidence = model_predictions(fitted_parameters)
fitted_total_rate = model_total_incidence_rate(fitted_incidence)

population = get_population(EQUILIBRIUM_YEAR)
population_by_group = aggregate_age_groups(
    population,
    CALIBRATION_AGE_BREAKS,
)

fitted_incidence_rates =
    1e5 .* fitted_incidence ./ population_by_group

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
        "  ",
        CALIBRATION_AGE_LABELS[g],
        ": cases = ",
        round(fitted_incidence[g], digits = 2),
        ", rate = ",
        round(fitted_incidence_rates[g], digits = 2),
        " per 100,000",
    )
end



# ---------------------------------------------------------------------------
# Notification-count diagnostics
# ---------------------------------------------------------------------------

# Equal-tailed predictive interval under the fitted negative-binomial model.
# Var(Y) = mu + sigma_obs^2 * mu^2.

function observation_interval(
    mu,
    sigma_obs,
    tau_year;
    level = 0.95,
    quadrature_order = 50,
)
    phi = inv(sigma_obs^2)

    nodes, weights =
        TBKiribatiJulia.normal_quadrature(quadrature_order)

    multipliers = exp.(
        tau_year .* nodes .- tau_year^2 / 2
    )

    distributions = [
        NegativeBinomial(
            phi,
            phi / (phi + mu * m),
        )
        for m in multipliers
    ]

    function mixture_cdf(y)
        return sum(
            w * cdf(dist, y)
            for (w, dist) in zip(weights, distributions)
        )
    end

    function mixture_quantile(p)
        p <= 0 && return 0

        hi = max(1, ceil(Int, mu))

        while mixture_cdf(hi) < p
            hi *= 2
        end

        lo = 0

        while lo < hi
            mid = lo + (hi - lo) ÷ 2

            if mixture_cdf(mid) >= p
                hi = mid
            else
                lo = mid + 1
            end
        end

        return lo
    end

    alpha = 1 - level

    return (
        mixture_quantile(alpha / 2),
        mixture_quantile(1 - alpha / 2),
    )
end


println()
println("Notification count fit (95% predictive intervals):")

n_outside = 0
n_observations = 0

for observation in NOTIFICATION_OBSERVATIONS

    detection = age_specific_detection(
        fitted_incidence,
        observation.overall_detection,
        RELATIVE_DETECTION,
    )

    expected = expected_notification_counts(
        fitted_incidence,
        observation;
        relative_detection = RELATIVE_DETECTION,
    )

    isnothing(detection) &&
        error("Invalid fitted detection probabilities for $(observation.year)")

    isnothing(expected) &&
        error("Invalid fitted expected counts for $(observation.year)")

    println()
    println(
        observation.year,
        "  overall detection = ",
        round(observation.overall_detection, digits = 3),
        "  observed total = ",
        sum(observation.counts),
        "  expected total = ",
        round(sum(expected), digits = 1),
    )

    for i in eachindex(observation.labels)

        observed = observation.counts[i]
        mu = expected[i]

        lo, hi = observation_interval(
    mu,
    fitted_sigma_obs,
    fitted_tau_year,
)

        outside = observed < lo || observed > hi

global n_observations += 1
global n_outside += outside

        println(
            "  ",
            rpad(observation.labels[i], 6),
            ": observed = ",
            lpad(observed, 4),
            ", expected = ",
            lpad(round(mu, digits = 1), 6),
            ", 95% PI = [",
            lpad(lo, 3),
            ", ",
            lpad(hi, 3),
            "]",
            outside ? "  *" : "",
        )
    end
end

println()
println("Predictive interval coverage:")
println(
    "  Outside 95% PI: ",
    n_outside,
    " / ",
    n_observations,
    " (",
    round(100 * n_outside / n_observations, digits = 1),
    "%)",
)
println("  * indicates an observation outside its 95% predictive interval")



# ---------------------------------------------------------------------------
# Detection probabilities
# ---------------------------------------------------------------------------

println()
println("Age-specific detection probabilities:")

for observation in NOTIFICATION_OBSERVATIONS

    detection = age_specific_detection(
        fitted_incidence,
        observation.overall_detection,
        RELATIVE_DETECTION,
    )

    isnothing(detection) &&
        error("Invalid fitted detection probabilities for $(observation.year)")

    println()
    println(
        observation.year,
        "  overall = ",
        round(observation.overall_detection, digits = 3),
    )

    for g in eachindex(CALIBRATION_AGE_LABELS)
        println(
            "  ",
            rpad(CALIBRATION_AGE_LABELS[g], 6),
            " = ",
            round(detection[g], digits = 3),
        )
    end
end




# ---------------------------------------------------------------------------
# Parametric bootstrap under the extended observation model
# ---------------------------------------------------------------------------

using Random
using Statistics
using Distributions

function extended_temporal_bootstrap(
    fitted_incidence,
    sigma_obs,
    tau_year,
    observations,
    relative_detection;
    n_boot = 10_000,
    seed = 12345,
)
    rng = MersenneTwister(seed)
    phi = inv(sigma_obs^2)

    selected = sort(collect(observations), by = x -> x.year)
    years = [obs.year for obs in selected]
    n_years = length(selected)

    expected = [
        expected_notification_counts(
            fitted_incidence,
            obs;
            relative_detection = relative_detection,
        )
        for obs in selected
    ]

    any(isnothing, expected) &&
        error("Invalid expected counts")

    observed_totals = [
        sum(obs.counts) for obs in selected
    ]

    expected_totals = [
        sum(mu) for mu in expected
    ]

    # Longest run of annual residuals of the same sign.
    function longest_sign_run(x)
        longest = 0
        current = 0
        previous = 0

        for r in x
            s = sign(r)
            current = s == 0 ? 0 :
                s == previous ? current + 1 : 1
            longest = max(longest, current)
            previous = s
        end

        return longest
    end

    function temporal_stats(totals)
        r = totals .- expected_totals
        return (
            longest_run = longest_sign_run(r),
            max_deviation = maximum(abs.(r)),
            residual_sd = std(r),
        )
    end

    observed_stats = temporal_stats(observed_totals)

    # Use consistent age categories for correlation analysis.
    reference_labels = collect(selected[1].labels)

    common_indices = findall(
        obs -> collect(obs.labels) == reference_labels,
        selected,
    )

    n_common = length(common_indices)
    n_groups = length(reference_labels)

    n_common >= 3 ||
        error("Insufficient years for correlation analysis")

    function standardised_matrix(counts)
        R = Matrix{Float64}(undef, n_common, n_groups)

        for (i, t) in enumerate(common_indices)
            for a in 1:n_groups
                mu = expected[t][a]

                # Marginal variance of the extended model.
                v = mu +
                    mu^2 * (
                        (1 + inv(phi)) * exp(tau_year^2) - 1
                    )

                R[i, a] = (counts[t][a] - mu) / sqrt(v)
            end
        end

        return R
    end

    function mean_pairwise_correlation(R)
        C = cor(R)
        n = size(C, 1)

        return sum(
            C[i, j]
            for i in 1:n-1
            for j in i+1:n
        ) / (n * (n - 1) / 2)
    end

    observed_counts = [collect(obs.counts) for obs in selected]

    observed_correlation = mean_pairwise_correlation(
        standardised_matrix(observed_counts)
    )

    boot_runs = Vector{Int}(undef, n_boot)
    boot_max = Vector{Float64}(undef, n_boot)
    boot_sd = Vector{Float64}(undef, n_boot)
    boot_corr = Vector{Float64}(undef, n_boot)

    simulated = [zeros(Int, length(mu)) for mu in expected]
    simulated_totals = zeros(Int, n_years)

    for b in 1:n_boot

        for t in 1:n_years

            # One shared random effect per year.
            u = tau_year * randn(rng)
            multiplier = exp(u - tau_year^2 / 2)

            for a in eachindex(expected[t])
                mu = expected[t][a] * multiplier

                dist = NegativeBinomial(
                    phi,
                    phi / (phi + mu),
                )

                simulated[t][a] = rand(rng, dist)
            end

            simulated_totals[t] = sum(simulated[t])
        end

        stats = temporal_stats(simulated_totals)

        boot_runs[b] = stats.longest_run
        boot_max[b] = stats.max_deviation
        boot_sd[b] = stats.residual_sd

        boot_corr[b] = mean_pairwise_correlation(
            standardised_matrix(simulated)
        )
    end

    upper_p(obs, draws) =
        (1 + count(x -> x >= obs, draws)) / (length(draws) + 1)

    # Two-sided correlation check, using empirical tails.
    lower_corr = (
        1 + count(x -> x <= observed_correlation, boot_corr)
    ) / (n_boot + 1)

    upper_corr = upper_p(observed_correlation, boot_corr)

    p_corr = min(1.0, 2 * min(lower_corr, upper_corr))

    println()
    println("Extended-model parametric bootstrap")
    println("  Replicates: ", n_boot)
    println("  sigma_obs: ", round(sigma_obs, digits = 4))
    println("  tau_year: ", round(tau_year, digits = 4))
    println()

    for (label, value, draws) in (
        ("Longest residual run",
            observed_stats.longest_run, boot_runs),
        ("Maximum annual deviation",
            observed_stats.max_deviation, boot_max),
        ("Annual residual SD",
            observed_stats.residual_sd, boot_sd),
    )
        println(
            rpad(label, 32),
            "observed = ",
            round(value, digits = 3),
            "  p = ",
            round(upper_p(value, draws), digits = 4),
        )
    end

    println(
        rpad("Mean age-group correlation", 32),
        "observed = ",
        round(observed_correlation, digits = 3),
        "  two-sided p = ",
        round(p_corr, digits = 4),
    )

    return (
        observed_stats = observed_stats,
        observed_correlation = observed_correlation,
        boot_runs = boot_runs,
        boot_max = boot_max,
        boot_sd = boot_sd,
        boot_corr = boot_corr,
        p_correlation = p_corr,
    )
end


extended_bootstrap_result = extended_temporal_bootstrap(
    fitted_incidence,
    fitted_sigma_obs,
    fitted_tau_year,
    NOTIFICATION_OBSERVATIONS,
    RELATIVE_DETECTION;
    n_boot = 10_000,
    seed = 12345,
)




# ---------------------------------------------------------------------------
# Residual correlation diagnostics
# ---------------------------------------------------------------------------

using Random
using Statistics
using LinearAlgebra
using Distributions

function residual_correlation_diagnostics(
    fitted_incidence,
    fitted_sigma_obs,
    observations,
    relative_detection;
    n_boot = 10_000,
    seed = 12345,
)

    rng = MersenneTwister(seed)
    phi = inv(fitted_sigma_obs^2)

    # Retain years with the standard eight age groups.
    reference_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+",
    ]

    selected = filter(
        obs -> collect(obs.labels) == reference_labels,
        observations,
    )

    sort!(selected, by = obs -> obs.year)

    years = [obs.year for obs in selected]
    n_years = length(years)
    n_groups = length(reference_labels)

    n_years >= 3 || error("Insufficient years for correlation analysis")

    # Observed and expected notification counts.
    observed = Matrix{Float64}(undef, n_years, n_groups)
    expected = Matrix{Float64}(undef, n_years, n_groups)

    for (t, obs) in enumerate(selected)

        mu = expected_notification_counts(
            fitted_incidence,
            obs;
            relative_detection = relative_detection,
        )

        isnothing(mu) &&
            error("Invalid expected counts for $(obs.year)")

        observed[t, :] .= obs.counts
        expected[t, :] .= mu
    end

    # NB2 standard deviations.
    sd_expected = sqrt.(
        expected .+ expected.^2 ./ phi
    )

    # Pearson standardised residuals.
    residuals = (observed .- expected) ./ sd_expected

    # Pairwise correlation matrix across years.
    correlation_matrix = cor(residuals)

    # Mean off-diagonal correlation.
    function mean_pairwise_correlation(R)
        n = size(R, 1)
        return sum(
            R[i, j]
            for i in 1:n-1
            for j in i+1:n
        ) / (n * (n - 1) / 2)
    end

    # Lag-1 autocorrelation of annual total residuals.
    function lag1_correlation(x)
        return cor(
            view(x, 1:length(x)-1),
            view(x, 2:length(x)),
        )
    end

    annual_residuals = vec(
        sum(observed .- expected, dims = 2)
    )

    observed_mean_correlation =
        mean_pairwise_correlation(correlation_matrix)

    observed_lag1 =
        lag1_correlation(annual_residuals)

    # Negative-binomial distributions.
    distributions = [
        NegativeBinomial(
            phi,
            phi / (phi + expected[t, a]),
        )
        for t in 1:n_years, a in 1:n_groups
    ]

    boot_mean_correlation = Vector{Float64}(undef, n_boot)
    boot_lag1 = Vector{Float64}(undef, n_boot)

    simulated = similar(observed)
    simulated_residuals = similar(residuals)

    for b in 1:n_boot

        for t in 1:n_years, a in 1:n_groups
            simulated[t, a] = rand(
                rng,
                distributions[t, a],
            )
        end

        simulated_residuals .=
            (simulated .- expected) ./ sd_expected

        R = cor(simulated_residuals)

        boot_mean_correlation[b] =
            mean_pairwise_correlation(R)

        simulated_annual_residuals = vec(
            sum(simulated .- expected, dims = 2)
        )

        boot_lag1[b] =
            lag1_correlation(simulated_annual_residuals)
    end

    # Upper-tail p-value for positive age-group correlation.
    p_mean_correlation = (
        1 + count(
            x -> x >= observed_mean_correlation,
            boot_mean_correlation,
        )
    ) / (n_boot + 1)

    # Two-sided bootstrap p-value for temporal autocorrelation,
    # using the simulated null distribution rather than symmetry.
    p_lag1 = (
        1 + count(
            x -> abs(x) >= abs(observed_lag1),
            boot_lag1,
        )
    ) / (n_boot + 1)

    println()
    println("Residual correlation diagnostics")
    println("  Years: ", first(years), "–", last(years))
    println("  Age groups: ", n_groups)
    println("  Bootstrap replicates: ", n_boot)

    println()
    println("Age-specific residual correlation matrix:")

    print(rpad("Age", 9))
    for label in reference_labels
        print(lpad(label, 9))
    end
    println()

    for i in 1:n_groups
        print(rpad(reference_labels[i], 9))

        for j in 1:n_groups
            print(
                lpad(
                    round(correlation_matrix[i, j], digits = 2),
                    9,
                )
            )
        end

        println()
    end

    println()
    println(
        "Mean pairwise age-group correlation = ",
        round(observed_mean_correlation, digits = 4),
    )
    println(
        "Bootstrap p-value (positive correlation) = ",
        round(p_mean_correlation, digits = 4),
    )

    println()
    println(
        "Lag-1 annual residual autocorrelation = ",
        round(observed_lag1, digits = 4),
    )
    println(
        "Bootstrap p-value (two-sided) = ",
        round(p_lag1, digits = 4),
    )

    return (
        years = years,
        labels = reference_labels,
        residuals = residuals,
        correlation_matrix = correlation_matrix,
        annual_residuals = annual_residuals,
        observed_mean_correlation = observed_mean_correlation,
        observed_lag1 = observed_lag1,
        boot_mean_correlation = boot_mean_correlation,
        boot_lag1 = boot_lag1,
        p_mean_correlation = p_mean_correlation,
        p_lag1 = p_lag1,
    )
end


# Run diagnostics.
correlation_result = residual_correlation_diagnostics(
    fitted_incidence,
    fitted_sigma_obs,
    NOTIFICATION_OBSERVATIONS,
    RELATIVE_DETECTION;
    n_boot = 10_000,
    seed = 12345,
)
