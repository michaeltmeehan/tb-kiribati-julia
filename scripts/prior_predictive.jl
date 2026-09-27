using TBKiribatiJulia

using DataFrames
using Distributions
using Random
using Statistics
using CairoMakie
using CSV
using StatsBase


# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------

# These are deliberately provisional priors.
#
# LogNormal(log(m), sigma) has median m, so the values below are centred
# approximately on the existing model/default starting values rather than
# being derived from the fitted MLE.

const PRIORS = (
    beta =
        LogNormal(log(1.0), 0.5),

    progression_child =
        LogNormal(log(2.4), 0.7),

    progression_5_14 =
        LogNormal(log(0.05), 0.8),

    progression_15_64 =
        LogNormal(log(0.2), 0.8),

    progression_65_plus =
        LogNormal(log(0.4), 0.7),
)


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

const CALIBRATION_AGE_BREAKS = [
    0,
    5,
    15,
    25,
    35,
    45,
    55,
    65,
    96,
]

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

const FIXED_PARAMETERS = (
    infectiousness_weights = (0.2, 0.5, 0.4, 1.0),
)

const TINIT = 1800.0
const TFINAL = 2024.0


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------

function sample_prior(rng, priors::NamedTuple)

    parameter_names = keys(priors)

    values = Tuple(
        rand(rng, distribution)
        for distribution in priors
    )

    return NamedTuple{parameter_names}(values)
end


# ---------------------------------------------------------------------------
# Prior predictive simulation
# ---------------------------------------------------------------------------

function run_prior_predictive(
    n_draws;
    seed = 1234,
)

    rng = MersenneTwister(seed)

    results = DataFrame()

    for draw in 1:n_draws

        θ = sample_prior(rng, PRIORS)

        try

            incidence =
                model_incidence_by_age_group(
                    θ;
                    age_breaks =
                        CALIBRATION_AGE_BREAKS,

                    fixed_parameters =
                        FIXED_PARAMETERS,

                    tinit =
                        TINIT,

                    tfinal =
                        TFINAL,
                )

            population =
    get_population(EQUILIBRIUM_YEAR)

population_by_group =
    aggregate_age_groups(
        population,
        CALIBRATION_AGE_BREAKS,
    )

incidence_rates =
    1e5 .* incidence ./ population_by_group

    relative_detection = [
    1.0,   # 0-4
    1.0,   # 5-14
    1.0,   # 15-24
    1.0,   # 25-34
    1.0,   # 35-44
    1.0,   # 45-54
    1.0,   # 55-64
    1.0,   # 65+
]

notification_weights =
    relative_detection .* incidence

notification_proportions =
    notification_weights ./ sum(notification_weights)

            total_rate =
                model_total_incidence_rate(
                    incidence,
                )

            row = merge(
    (
        draw = draw,
        valid = true,
        total_incidence_rate = total_rate,
    ),
    θ,
    NamedTuple{
        Tuple(
            Symbol(
                "incidence_",
                replace(label, "-" => "_", "+" => "plus"),
            )
            for label in CALIBRATION_AGE_LABELS
        )
    }(
        Tuple(incidence)
    ),
    NamedTuple{
        Tuple(
            Symbol(
                "incidence_rate_",
                replace(label, "-" => "_", "+" => "plus"),
            )
            for label in CALIBRATION_AGE_LABELS
        )
    }(
        Tuple(incidence_rates)
    ),
    NamedTuple{
    Tuple(
        Symbol(
            "notification_prop_",
            replace(label, "-" => "_", "+" => "plus"),
        )
        for label in CALIBRATION_AGE_LABELS
    )
}(
    Tuple(notification_proportions)
),
)

            push!(
                results,
                row;
                cols = :union,
            )

        catch err

            @warn(
                "Prior predictive simulation failed",
                draw = draw,
                parameters = θ,
                exception = err,
            )

            row = merge(
                (
                    draw = draw,
                    valid = false,
                    total_incidence_rate = NaN,
                ),
                θ,
            )

            push!(
                results,
                row;
                cols = :union,
            )
        end
    end

    return results
end


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

results =
    run_prior_predictive(
        500,
    )


# ---------------------------------------------------------------------------
# Basic diagnostics
# ---------------------------------------------------------------------------

valid_results =
    results[results.valid .== true, :]


println()
println(
    "Valid simulations: ",
    nrow(valid_results),
    " / ",
    nrow(results),
)


println()
println("Prior predictive total incidence rate per 100,000:")

println(
    "  2.5%  = ",
    quantile(
        valid_results.total_incidence_rate,
        0.025,
    ),
)

println(
    "  25%   = ",
    quantile(
        valid_results.total_incidence_rate,
        0.25,
    ),
)

println(
    "  50%   = ",
    median(
        valid_results.total_incidence_rate,
    ),
)

println(
    "  75%   = ",
    quantile(
        valid_results.total_incidence_rate,
        0.75,
    ),
)

println(
    "  97.5% = ",
    quantile(
        valid_results.total_incidence_rate,
        0.975,
    ),
)

# ---------------------------------------------------------------------------
# Extreme prior-predictive draws
# ---------------------------------------------------------------------------

sorted_results =
    sort(
        valid_results,
        :total_incidence_rate,
    )


println()
println("Lowest-incidence prior draws:")

show(
    first(sorted_results, 10)[
        :,
        [
            :total_incidence_rate,
            keys(PRIORS)...,
        ],
    ],
    allrows = true,
    allcols = true,
)

println()


println()
println("Highest-incidence prior draws:")

show(
    last(sorted_results, 10)[
        :,
        [
            :total_incidence_rate,
            keys(PRIORS)...,
        ],
    ],
    allrows = true,
    allcols = true,
)

println()


# ---------------------------------------------------------------------------
# Parameter-output associations
# ---------------------------------------------------------------------------

println()
println("Spearman correlations with total incidence:")

for parameter in keys(PRIORS)

    ρ = corspearman(
        valid_results[!, parameter],
        valid_results.total_incidence_rate,
    )

    println(
        rpad(string(parameter), 25),
        " = ",
        round(ρ, digits = 3),
    )
end


# ---------------------------------------------------------------------------
# Age-specific prior predictive summaries
# ---------------------------------------------------------------------------

println()
println("Prior predictive incidence rate per 100,000 by age group:")

for label in CALIBRATION_AGE_LABELS

    column =
    Symbol(
        "incidence_rate_",
        replace(label, "-" => "_", "+" => "plus"),
    )

    values = valid_results[!, column]

    println()
    println(label)

    println(
        "  2.5%  = ",
        round(quantile(values, 0.025), digits = 2),
    )

    println(
        "  25%   = ",
        round(quantile(values, 0.25), digits = 2),
    )

    println(
        "  50%   = ",
        round(median(values), digits = 2),
    )

    println(
        "  75%   = ",
        round(quantile(values, 0.75), digits = 2),
    )

    println(
        "  97.5% = ",
        round(quantile(values, 0.975), digits = 2),
    )
end


# ---------------------------------------------------------------------------
# Parameter associations with age-specific incidence rates
# ---------------------------------------------------------------------------

println()
println("Spearman correlations with age-specific incidence rates:")

for parameter in keys(PRIORS)

    println()
    println(parameter)

    for label in CALIBRATION_AGE_LABELS

        column =
            Symbol(
                "incidence_rate_",
                replace(label, "-" => "_", "+" => "plus"),
            )

        ρ =
            corspearman(
                valid_results[!, parameter],
                valid_results[!, column],
            )

        println(
            "  ",
            rpad(label, 8),
            " = ",
            round(ρ, digits = 3),
        )
    end
end


# ---------------------------------------------------------------------------
# Prior predictive check against WHO total incidence estimates
# ---------------------------------------------------------------------------

incidence_data = CSV.read(
    joinpath(@__DIR__, "..", "data", "tb_incidence.csv"),
    DataFrame,
)

incidence_data = incidence_data[
    (incidence_data.incidence_rate .> 0) .&
    (incidence_data.incidence_lower .> 0) .&
    (incidence_data.incidence_upper .> 0),
    :,
]


# ---------------------------------------------------------------------------
# Prior predictive coverage of observed incidence range
# ---------------------------------------------------------------------------

observed_lower =
    minimum(incidence_data.incidence_lower)

observed_upper =
    maximum(incidence_data.incidence_upper)

prior_rates =
    valid_results.total_incidence_rate


p_below =
    mean(prior_rates .< observed_lower)

p_within =
    mean(
        (prior_rates .>= observed_lower) .&
        (prior_rates .<= observed_upper)
    )

p_above =
    mean(prior_rates .> observed_upper)


println()
println("Prior predictive mass relative to WHO incidence range:")

println(
    "  Below observed uncertainty range = ",
    round(100 * p_below, digits = 1),
    "%"
)

println(
    "  Within observed uncertainty range = ",
    round(100 * p_within, digits = 1),
    "%"
)

println(
    "  Above observed uncertainty range = ",
    round(100 * p_above, digits = 1),
    "%"
)

println()
println(
    "WHO uncertainty envelope = ",
    round(observed_lower, digits = 1),
    "–",
    round(observed_upper, digits = 1),
    " per 100,000"
)


log_prior_incidence =
    log10.(valid_results.total_incidence_rate)

log_who =
    log10.(Float64.(incidence_data.incidence_rate))

log_who_lower =
    log10.(Float64.(incidence_data.incidence_lower))

log_who_upper =
    log10.(Float64.(incidence_data.incidence_upper))


fig = Figure(size = (900, 550))

ax = Axis(
    fig[1, 1],
    xlabel = "log10 total TB incidence rate per 100,000",
    ylabel = "Prior predictive density",
)


hist!(
    ax,
    log_prior_incidence;
    bins = 40,
    normalization = :pdf,
)


# Put WHO estimates near the baseline.
y_who =
    zeros(length(log_who))

errorbars!(
    ax,
    log_who,
    y_who,
    log_who .- log_who_lower,
    log_who_upper .- log_who;
    direction = :x,
)

scatter!(
    ax,
    log_who,
    y_who,
)


output_dir =
    joinpath(
        @__DIR__,
        "..",
        "output",
    )

mkpath(output_dir)

save(
    joinpath(
        output_dir,
        "prior_predictive_total_incidence.png",
    ),
    fig,
)

fig