using TBKiribatiJulia
using DataFrames
using Plots


# Simulation settings ---------------------------------------------------------

burnin_start = 1800.0
projection_start = 2025.0
projection_end = 2100.0


# Baseline fitted parameters --------------------------------------------------

baseline = (
    beta = 1.1,
    progression_child = 0.5,
    progression_5_14 = 0.3,
    progression_15_64 = 0.15,
    progression_65_plus = 0.3,
)

sensitivity_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]


# Parameter constructor -------------------------------------------------------

function make_sensitivity_parameters(;
    beta = baseline.beta,
    progression_child = baseline.progression_child,
    progression_5_14 = baseline.progression_5_14,
    progression_15_64 = baseline.progression_15_64,
    progression_65_plus = baseline.progression_65_plus,
)
    return make_parameters(
        CONTACT;
        beta = beta,
        progression_child = progression_child,
        progression_5_14 = progression_5_14,
        progression_15_64 = progression_15_64,
        progression_65_plus = progression_65_plus,
        infectiousness_weights = (0.2, 0.5, 0.4, 1.0),

        # TB-related mortality
        pct_neg_tx_death = 0.4,
        disease_mortality_clin_lowinf = 0.025,
        disease_mortality_clin_inf = 0.4,
    )
end


# Parameters to perturb -------------------------------------------------------

sensitivity_parameters = (
    beta = baseline.beta,
    progression_child = baseline.progression_child,
    progression_5_14 = baseline.progression_5_14,
    progression_15_64 = baseline.progression_15_64,
    progression_65_plus = baseline.progression_65_plus,
)


# Run one-at-a-time sensitivity analysis -------------------------------------

results = DataFrame(
    parameter = String[],
    multiplier = Float64[],
    parameter_value = Float64[],
    year = Float64[],
    incidence_log_ratio = Float64[],
    incidence_relative_difference = Float64[],
    mortality_log_ratio = Float64[],
    mortality_relative_difference = Float64[],
)

for parameter in keys(sensitivity_parameters)

    baseline_value = sensitivity_parameters[parameter]

    for multiplier in sensitivity_multipliers

        parameter_value = baseline_value * multiplier

        kwargs = NamedTuple{(parameter,)}((parameter_value,))
        params = make_sensitivity_parameters(; kwargs...)

        result = run_demography_comparison(
            params;
            burnin_start = burnin_start,
            projection_start = projection_start,
            projection_end = projection_end,
        )

        comparison = comparison_summary(result)

        for i in eachindex(comparison.years)
            push!(results, (
                parameter = String(parameter),
                multiplier = multiplier,
                parameter_value = parameter_value,
                year = comparison.years[i],
                incidence_log_ratio =
                    comparison.incidence_log_ratio[i],
                incidence_relative_difference =
                    comparison.incidence_relative_difference[i],
                mortality_log_ratio =
                    comparison.mortality_log_ratio[i],
                mortality_relative_difference =
                    comparison.mortality_relative_difference[i],
            ))
        end
    end
end


results

summary_years = [2050.0, 2075.0, 2100.0]

endpoint_results = filter(
    row -> row.year in summary_years,
    results,
)

sort!(
    endpoint_results,
    [:parameter, :year, :multiplier],
)

endpoint_results


endpoint_2100 = filter(
    row -> row.year == 2100.0,
    endpoint_results,
)

select!(
    endpoint_2100,
    :parameter,
    :multiplier,
    :parameter_value,
    :incidence_relative_difference,
    :mortality_relative_difference,
)

endpoint_2100


p_incidence = Plots.plot(
    xlabel = "Parameter multiplier",
    ylabel = "2100 incidence difference (%)",
    legend = :topright,
)

for parameter in unique(endpoint_2100.parameter)
    df = filter(
        row -> row.parameter == parameter,
        endpoint_2100,
    )

    Plots.plot!(
        p_incidence,
        df.multiplier,
        df.incidence_relative_difference;
        marker = :circle,
        linewidth = 2,
        label = parameter,
    )
end

p_incidence


p_mortality = Plots.plot(
    xlabel = "Parameter multiplier",
    ylabel = "2100 mortality difference (%)",
    legend = :topright,
)

for parameter in unique(endpoint_2100.parameter)
    df = filter(
        row -> row.parameter == parameter,
        endpoint_2100,
    )

    Plots.plot!(
        p_mortality,
        df.multiplier,
        df.mortality_relative_difference;
        marker = :circle,
        linewidth = 2,
        label = parameter,
    )
end

p_mortality