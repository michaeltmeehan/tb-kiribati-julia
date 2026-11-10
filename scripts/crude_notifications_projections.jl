# crude_notifications_projections.jl
#
# Kiribati TB: demographic counterfactual with fixed age-specific notification rates.
#
# Run after loading the existing project's `get_population(year)` and
# `aggregate_age_groups(population, age_breaks)` functions into the Julia session.
#  No mechanistic TB model is run here.
#
# Three fixed-rate specifications:
#   1. Observed 2023 rates
#   2. Pooled 2019-2023 rates: sum(cases) / sum(age-specific person-years)
#   3. Pooled 2013-2023 rates: same exposure-weighted estimator
#
# Each specification uses the *2023 population* as its counterfactual baseline.
# Comparisons for 2023-2100:
#   Ageing only:       2023 population total x future age proportions
#   Population only:   future population total x 2023 age proportions
#   Combined:          actual/projected future age-group populations
#
# Rates are expressed per 100,000 persons per year. Expected notifications
# are not forecasts of actual future TB notifications: epidemiological rates
# and ascertainment are deliberately held constant.

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using CairoMakie
using Printf

let
    data_dir = normpath(joinpath(@__DIR__, "..", "data"))
    output_dir = normpath(joinpath(@__DIR__, "..", "results", "notification_ageing"))
    mkpath(output_dir)

    notification_file = joinpath(data_dir, "tb_notifications_by_age.csv")
    isfile(notification_file) || error("Notification file not found: $notification_file")
    notifications = CSV.read(notification_file, DataFrame)

    age_breaks = [0, 5, 15, 25, 35, 45, 55, 65, 96]
    age_labels = ["0-4", "5-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    n_ages = length(age_labels)

    baseline_year = 2023
    final_year = 2050
    milestones = [2030, 2040, 2050]
    specifications = [
        ("2023", 2023:2023),
        ("2019-2023 pooled", 2019:2023),
        ("2013-2023 pooled", 2013:2023),
    ]
    primary_specification = "2019-2023 pooled"

    # Use the same demographic helpers and age breaks as the supplied script.
    # Passing Float64 mirrors its use of YEARS = 2013.:2023.
    function age_group_population(year::Int)
        grouped = Float64.(aggregate_age_groups(get_population(Float64(year)), age_breaks))
        length(grouped) == n_ages ||
            error("Population aggregation returned $(length(grouped)) groups in $year, expected $n_ages")
        all(isfinite, grouped) && all(grouped .> 0) ||
            error("Missing, nonfinite or nonpositive age-group population in year $year")
        return grouped
    end

    # Cache the population once per year (historical exposures + all projections).
    populations = Dict(year => age_group_population(year) for year in 2013:final_year)
    baseline_population = populations[baseline_year]
    baseline_total = sum(baseline_population)
    baseline_proportions = baseline_population ./ baseline_total

    # Read precisely one nonmissing observation per age and required year.
    # 2024 is never used to estimate rates (its childhood bands are incomplete).
    function observed_cases(year::Int, age_group::String)
        selected = notifications[
            (notifications.year .== year) .& (notifications.age_group .== age_group),
            :cases,
        ]
        length(selected) == 1 ||
            error("Expected exactly one notification record for $year / $age_group, found $(length(selected))")
        value = only(selected)
        ismissing(value) && error("Missing notification count for $year / $age_group")
        value = Float64(value)
        isfinite(value) && value >= 0 ||
            error("Invalid notification count for $year / $age_group: $value")
        return value
    end

    # Properly pooled rates are weighted by the population exposure in each
    # age group. For annual population estimates, the summed populations serve
    # as the approximate total person-years of exposure.
    rates_by_specification = Dict{String, Vector{Float64}}()
    for (specification, years) in specifications
        summed_cases = zeros(n_ages)
        summed_exposures = zeros(n_ages)
        for year in years
            summed_exposures .+= populations[year]
            for (i, age_group) in enumerate(age_labels)
                summed_cases[i] += observed_cases(year, age_group)
            end
        end
        rates_by_specification[specification] = 1e5 .* summed_cases ./ summed_exposures
    end

    rate_table = DataFrame(age_group = age_labels)
    for (specification, _) in specifications
        rate_table[!, specification] = rates_by_specification[specification]
    end
    CSV.write(joinpath(output_dir, "fixed_age_specific_rates.csv"), rate_table)

    # Population-demography table, useful for checking the underlying change.
    population_table = DataFrame(year = Int[], total_population = Float64[], share_65_plus = Float64[])
    for year in baseline_year:final_year
        pop = populations[year]
        push!(population_table, (year, sum(pop), pop[end] / sum(pop)))
    end
    CSV.write(joinpath(output_dir, "projected_population_composition.csv"), population_table)

    # Comprehensive annual projection table.
    projection = DataFrame(
        specification = String[],
        year = Int[],
        total_population = Float64[],
        share_65_plus = Float64[],
        weighted_notification_rate = Float64[],
        baseline_expected_cases = Float64[],
        ageing_only_expected_cases = Float64[],
        population_only_expected_cases = Float64[],
        combined_expected_cases = Float64[],
        ageing_pct = Float64[],
        population_growth_pct = Float64[],
        combined_pct = Float64[],
        interaction_pct = Float64[],
    )

    for (specification, _) in specifications
        rates = rates_by_specification[specification]
        baseline_weighted_rate = dot(baseline_proportions, rates)
        baseline_cases = baseline_total * baseline_weighted_rate / 1e5
        baseline_cases > 0 || error("Nonpositive baseline notifications for $specification")

        for year in baseline_year:final_year
            pop = populations[year]
            total = sum(pop)
            proportions = pop ./ total
            weighted_rate = dot(proportions, rates)

            ageing_only_cases = baseline_total * weighted_rate / 1e5
            population_only_cases = total * baseline_weighted_rate / 1e5
            combined_cases = total * weighted_rate / 1e5

            ageing_pct = 100 * (ageing_only_cases / baseline_cases - 1)
            population_growth_pct = 100 * (population_only_cases / baseline_cases - 1)
            combined_pct = 100 * (combined_cases / baseline_cases - 1)
            interaction_pct = combined_pct - ageing_pct - population_growth_pct

            push!(projection, (
                specification = specification,
                year = year,
                total_population = total,
                share_65_plus = proportions[end],
                weighted_notification_rate = weighted_rate,
                baseline_expected_cases = baseline_cases,
                ageing_only_expected_cases = ageing_only_cases,
                population_only_expected_cases = population_only_cases,
                combined_expected_cases = combined_cases,
                ageing_pct = ageing_pct,
                population_growth_pct = population_growth_pct,
                combined_pct = combined_pct,
                interaction_pct = interaction_pct,
            ))
        end
    end

    # The three scenarios must agree exactly (up to floating-point rounding) in 2023.
    for row in eachrow(projection[projection.year .== baseline_year, :])
        @assert isapprox(row.ageing_pct, 0.0; atol=1e-10)
        @assert isapprox(row.population_growth_pct, 0.0; atol=1e-10)
        @assert isapprox(row.combined_pct, 0.0; atol=1e-10)
    end

    CSV.write(joinpath(output_dir, "annual_demographic_counterfactuals.csv"), projection)

    # Main summary: year 2023 for reference, and each requested horizon.
    summary = projection[in.(projection.year, Ref([baseline_year; milestones])), :]
    sort!(summary, [:specification, :year])
    CSV.write(joinpath(output_dir, "milestone_demographic_counterfactuals.csv"), summary)

    # Per-age contribution to the ageing-only change in expected notifications.
    # These contributions sum to ageing_only_expected_cases - baseline_expected_cases.
    contributions = DataFrame(
        specification = String[],
        year = Int[],
        age_group = String[],
        baseline_share = Float64[],
        projected_share = Float64[],
        fixed_notification_rate = Float64[],
        ageing_only_case_change = Float64[],
        ageing_pct_point_contribution = Float64[],
    )
    for (specification, _) in specifications
        rates = rates_by_specification[specification]
        baseline_cases = baseline_total * dot(baseline_proportions, rates) / 1e5
        for year in milestones
            future_proportions = populations[year] ./ sum(populations[year])
            for i in eachindex(age_labels)
                case_change = baseline_total / 1e5 *
                              (future_proportions[i] - baseline_proportions[i]) * rates[i]
                push!(contributions, (
                    specification, year, age_labels[i], baseline_proportions[i],
                    future_proportions[i], rates[i], case_change,
                    100 * case_change / baseline_cases,
                ))
            end
            projected = only(eachrow(projection[
                (projection.specification .== specification) .& (projection.year .== year), :,
            ]))
            sum_change = sum(contributions[
                (contributions.specification .== specification) .& (contributions.year .== year),
                :ageing_only_case_change,
            ])
            @assert isapprox(sum_change,
                projected.ageing_only_expected_cases - projected.baseline_expected_cases;
                rtol=1e-10, atol=1e-8)
        end
    end
    CSV.write(joinpath(output_dir, "age_group_contributions.csv"), contributions)

    # Figure 1: isolated effect of demographic ageing, for all rate specifications.
    colors = [:royalblue, :darkorange, :seagreen]
    fig1 = Figure(size = (1050, 640))
    ax1 = Axis(
        fig1[1, 1],
        xlabel = "Year",
        ylabel = "Change in expected notifications (%)",
        title = "Kiribati TB: effect of changing age composition alone",
        subtitle = "Total population fixed at 2023; age-specific notification rates held constant",
        xticks = [2023, 2030, 2040, 2050],
    )
    hlines!(ax1, [0.0], color = (:gray, 0.6), linestyle = :dash)
    for (i, (specification, _)) in enumerate(specifications)
        sub = projection[projection.specification .== specification, :]
        lines!(ax1, sub.year, sub.ageing_pct,
               linewidth = 3, color = colors[i], label = specification)
        points = [only(sub[sub.year .== year, :ageing_pct]) for year in milestones]
        CairoMakie.scatter!(ax1, milestones, points, color = colors[i], markersize = 12)
    end
    axislegend(ax1, position = :lt)
    save(joinpath(output_dir, "ageing_only_three_specifications.png"), fig1, px_per_unit = 2)

    # Figure 2: separate population-size and age-composition effects, using
    # the recent (2019-2023) pooled rate specification as the primary analysis.
    sub = projection[projection.specification .== primary_specification, :]
    fig2 = Figure(size = (1050, 640))
    ax2 = Axis(
        fig2[1, 1],
        xlabel = "Year",
        ylabel = "Change in expected notifications (%)",
        title = "Kiribati TB: demographic decomposition",
        subtitle = "Fixed 2019-2023 pooled age-specific rates; 2023 population baseline",
        xticks = [2023, 2030, 2040, 2050],
    )
    hlines!(ax2, [0.0], color = (:gray, 0.6), linestyle = :dash)
    scenarios = [
        ("Age composition only", :ageing_pct, :royalblue),
        ("Population size only", :population_growth_pct, :darkorange),
        ("Combined demographic effect", :combined_pct, :seagreen),
    ]
    for (label, column, color) in scenarios
        lines!(ax2, sub.year, sub[!, column], linewidth = 3, color = color, label = label)
        points = [only(sub[sub.year .== year, column]) for year in milestones]
        CairoMakie.scatter!(ax2, milestones, points, color = color, markersize = 12)
    end
    axislegend(ax2, position = :lt)
    save(joinpath(output_dir, "demographic_decomposition_recent_pooled.png"), fig2, px_per_unit = 2)

    # Print requested horizon results without asserting that any increase must occur.
    println("\nTB notification demographic counterfactual (baseline: $baseline_year)")
    println("Age-specific rates held fixed; all percentage changes are relative to")
    println("the specification-specific expected notifications in $baseline_year.")
    @printf("2023 population: %.0f; share aged 65+: %.2f%%\n",
            baseline_total, 100 * baseline_proportions[end])
    println()
    for (specification, _) in specifications
        println(specification)
        @printf("  %-6s %12s %12s %12s %12s %15s\n",
                "Year", "Ageing %", "Size %", "Total %", "Interact %", "Expected cases")
        for year in milestones
            row = only(eachrow(projection[
                (projection.specification .== specification) .& (projection.year .== year), :,
            ]))
            @printf("  %-6d %12.2f %12.2f %12.2f %12.2f %15.1f\n",
                    year, row.ageing_pct, row.population_growth_pct,
                    row.combined_pct, row.interaction_pct,
                    row.combined_expected_cases)
        end
        println()
    end
    println("CSV tables and PNG figures saved to: $output_dir")
end
