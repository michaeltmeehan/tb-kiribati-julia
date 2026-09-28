using TBKiribatiJulia

using DataFrames
using Distributions
using Random
using Statistics
using CairoMakie
using StatsBase

include("calibration_data.jl")


# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------

# Transmission priors are unchanged from the previous prior-predictive check.
const TRANSMISSION_PRIORS = (
    beta = LogNormal(log(0.5), 0.3),
    progression_child = LogNormal(log(1.0), 0.5),
    progression_5_14 = LogNormal(log(0.5), 0.5),
    progression_15_64 = LogNormal(log(0.5), 0.5),
    progression_65_plus = LogNormal(log(0.5), 0.5),
)

# Extra-Poisson coefficient of variation. Under the NB2 parameterisation,
# phi = 1 / sigma_obs^2 and Var(Y) = mu + mu^2 / phi.
const SIGMA_OBS_PRIOR = truncated(Normal(0.0, 0.5), 0.0, Inf)


function sample_prior(rng)
    names = keys(TRANSMISSION_PRIORS)
    values = Tuple(rand(rng, d) for d in TRANSMISSION_PRIORS)

    return (
        transmission = NamedTuple{names}(values),
        sigma_obs = rand(rng, SIGMA_OBS_PRIOR),
    )
end


# ---------------------------------------------------------------------------
# Prior predictive simulation
# ---------------------------------------------------------------------------

function simulate_notification_observation(
    rng,
    predicted_incidence,
    observation,
    sigma_obs,
)
    expected = expected_notification_counts(
        predicted_incidence,
        observation;
        relative_detection = RELATIVE_DETECTION,
    )

    isnothing(expected) && return nothing

    phi = inv(sigma_obs^2)
    replicated = Int[]

    for mu in expected
        dist = negative_binomial_mu_phi(mu, phi)
        isnothing(dist) && return nothing
        push!(replicated, rand(rng, dist))
    end

    return (
        expected = expected,
        replicated = replicated,
    )
end


function run_prior_predictive(
    n_draws;
    seed = 1234,
)
    rng = MersenneTwister(seed)

    parameter_results = DataFrame()
    notification_results = DataFrame()

    for draw in 1:n_draws
        prior_draw = sample_prior(rng)
        theta = prior_draw.transmission
        sigma_obs = prior_draw.sigma_obs

        try
            incidence = model_incidence_by_age_group(
                theta;
                age_breaks = CALIBRATION_AGE_BREAKS,
                fixed_parameters = FIXED_PARAMETERS,
                tinit = TINIT,
                tfinal = TFINAL,
            )

            if any(x -> !isfinite(x) || x <= 0, incidence)
                error("Invalid model-predicted incidence")
            end

            population = get_population(EQUILIBRIUM_YEAR)
            population_by_group = aggregate_age_groups(
                population,
                CALIBRATION_AGE_BREAKS,
            )

            incidence_rates = 1e5 .* incidence ./ population_by_group
            total_rate = model_total_incidence_rate(incidence)
            phi = inv(sigma_obs^2)

            parameter_row = merge(
                (
                    draw = draw,
                    valid = true,
                    sigma_obs = sigma_obs,
                    phi = phi,
                    total_incidence_rate = total_rate,
                ),
                theta,
                NamedTuple{
                    Tuple(
                        Symbol(
                            "incidence_rate_",
                            replace(label, "-" => "_", "+" => "plus"),
                        )
                        for label in CALIBRATION_AGE_LABELS
                    )
                }(Tuple(incidence_rates)),
            )

            push!(parameter_results, parameter_row; cols = :union)

            for observation in NOTIFICATION_OBSERVATIONS
                simulation = simulate_notification_observation(
                    rng,
                    incidence,
                    observation,
                    sigma_obs,
                )

                isnothing(simulation) &&
                    error("Invalid notification expectation")

                for j in eachindex(observation.labels)
                    push!(
                        notification_results,
                        (
                            draw = draw,
                            year = observation.year,
                            age_group = observation.labels[j],
                            overall_detection = observation.overall_detection,
                            observed = observation.counts[j],
                            expected = simulation.expected[j],
                            replicated = simulation.replicated[j],
                        ),
                    )
                end
            end

        catch err
            @warn(
                "Prior predictive simulation failed",
                draw = draw,
                parameters = theta,
                sigma_obs = sigma_obs,
                exception = err,
            )

            push!(
                parameter_results,
                merge(
                    (
                        draw = draw,
                        valid = false,
                        sigma_obs = sigma_obs,
                        phi = inv(sigma_obs^2),
                        total_incidence_rate = NaN,
                    ),
                    theta,
                );
                cols = :union,
            )
        end
    end

    return parameter_results, notification_results
end


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

parameter_results, notification_results = run_prior_predictive(500)
valid_results = parameter_results[parameter_results.valid .== true, :]

println()
println("Valid simulations: ", nrow(valid_results), " / ", nrow(parameter_results))


# ---------------------------------------------------------------------------
# Prior summaries
# ---------------------------------------------------------------------------

function print_quantiles(label, values)
    println()
    println(label)
    println("  2.5%  = ", round(quantile(values, 0.025), digits = 3))
    println("  25%   = ", round(quantile(values, 0.25), digits = 3))
    println("  50%   = ", round(median(values), digits = 3))
    println("  75%   = ", round(quantile(values, 0.75), digits = 3))
    println("  97.5% = ", round(quantile(values, 0.975), digits = 3))
end

print_quantiles(
    "Prior sigma_obs:",
    valid_results.sigma_obs,
)

print_quantiles(
    "Prior phi = 1 / sigma_obs^2:",
    valid_results.phi,
)

print_quantiles(
    "Prior predictive total incidence rate per 100,000:",
    valid_results.total_incidence_rate,
)


# ---------------------------------------------------------------------------
# Age-specific incidence summaries
# ---------------------------------------------------------------------------

println()
println("Prior predictive incidence rate per 100,000 by age group:")

for label in CALIBRATION_AGE_LABELS
    column = Symbol(
        "incidence_rate_",
        replace(label, "-" => "_", "+" => "plus"),
    )

    print_quantiles(label, valid_results[!, column])
end


# ---------------------------------------------------------------------------
# Parameter-output associations
# ---------------------------------------------------------------------------

println()
println("Spearman correlations with total incidence:")

for parameter in keys(TRANSMISSION_PRIORS)
    rho = corspearman(
        valid_results[!, parameter],
        valid_results.total_incidence_rate,
    )

    println(
        rpad(string(parameter), 25),
        " = ",
        round(rho, digits = 3),
    )
end


# ---------------------------------------------------------------------------
# Prior predictive notification checks
# ---------------------------------------------------------------------------

println()
println("Prior predictive notification counts:")

for observation in NOTIFICATION_OBSERVATIONS
    println()
    println(
        observation.year,
        "  (WHO overall detection = ",
        round(observation.overall_detection, digits = 3),
        ")",
    )

    for (j, label) in enumerate(observation.labels)
        rows = notification_results[
            (notification_results.year .== observation.year) .&
            (notification_results.age_group .== label),
            :,
        ]

        values = rows.replicated

        println(
            "  ",
            rpad(label, 8),
            " observed = ",
            observation.counts[j],
            ", prior predictive 95% = [",
            round(quantile(values, 0.025), digits = 1),
            ", ",
            round(quantile(values, 0.975), digits = 1),
            "]",
            ", median = ",
            round(median(values), digits = 1),
        )
    end
end


# ---------------------------------------------------------------------------
# Annual total-notification check
# ---------------------------------------------------------------------------

annual_predictive = combine(
    groupby(notification_results, [:draw, :year]),
    :replicated => sum => :replicated_total,
    :expected => sum => :expected_total,
    :observed => sum => :observed_total,
)

println()
println("Prior predictive annual total notifications:")

for year in CALIBRATION_YEARS
    rows = annual_predictive[annual_predictive.year .== year, :]

    println(
        "  ", year,
        ": observed = ", first(rows.observed_total),
        ", prior predictive 95% = [",
        round(quantile(rows.replicated_total, 0.025), digits = 1),
        ", ",
        round(quantile(rows.replicated_total, 0.975), digits = 1),
        "]",
        ", median = ",
        round(median(rows.replicated_total), digits = 1),
    )
end


# ---------------------------------------------------------------------------
# Plot annual total notifications
# ---------------------------------------------------------------------------

summary_rows = combine(
    groupby(annual_predictive, :year),
    :replicated_total => (x -> quantile(x, 0.025)) => :lower,
    :replicated_total => median => :median,
    :replicated_total => (x -> quantile(x, 0.975)) => :upper,
    :observed_total => first => :observed,
)

fig = Figure(size = (900, 550))
ax = Axis(
    fig[1, 1],
    xlabel = "Year",
    ylabel = "TB notifications",
)

band!(
    ax,
    summary_rows.year,
    summary_rows.lower,
    summary_rows.upper,
)

lines!(
    ax,
    summary_rows.year,
    summary_rows.median,
)

scatter!(
    ax,
    summary_rows.year,
    summary_rows.observed,
)

output_dir = joinpath(@__DIR__, "..", "output")
mkpath(output_dir)

save(
    joinpath(output_dir, "prior_predictive_notifications.png"),
    fig,
)

fig
