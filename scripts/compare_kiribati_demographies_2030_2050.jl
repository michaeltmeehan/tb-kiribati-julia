# compare_kiribati_demographies_2030_2050.jl
#
# Compare the projected Kiribati age distributions in 2030, 2040 and 2050.
# Produces overlaid, lightly filled single-year-age step histograms:
#   (1) age composition (% of population at each age), and
#   (2) absolute number of people at each age.
#
# Assumes get_population(year) is already defined in the Julia session.
# As in the TB model, the returned population is ordered by single-year age:
# index 1 is age 0, index 2 is age 1, etc.; its last element can be a
# top-coded oldest-age group (e.g. age 95+).
#
# Run from the project root after loading demographic utilities:
#   include("scripts/compare_kiribati_demographies_2030_2050.jl")
#
# Files saved to results/demographic_comparison/:
#   population_age_distributions_2030_2040_2050.png
#   population_age_distributions_2030_2040_2050.svg
#   population_age_summary.csv
#   population_by_single_year_age.csv

using CairoMakie
using CSV
using DataFrames
using Printf

let
    years = [2030, 2040, 2050]
    output_dir = normpath(joinpath(@__DIR__, "..", "results", "demographic_comparison"))
    mkpath(output_dir)

    # Read demographic projections, preserving single-year ages.
    populations = Dict{Int, Vector{Float64}}()
    for year in years
        raw = get_population(Float64(year))
        pop = Float64.(vec(raw))
        length(pop) >= 66 || error("Expected at least ages 0 to 65 in $year; received $(length(pop)) values")
        all(isfinite, pop) && all(pop .>= 0) && sum(pop) > 0 ||
            error("Nonfinite, negative or zero population encountered for year $year")
        populations[year] = pop
    end
    n_ages = length(populations[first(years)])
    all(length(populations[year]) == n_ages for year in years) ||
        error("Population vectors have different age-group lengths across years")
    top_age = n_ages - 1  # The last category may represent top_age+.

    # Population-weighted median, interpolating uniformly within a 1-year bin.
    # The top-coded final age group is treated as a 1-year bin for plotting;
    # this has negligible bearing on the median unless it falls in that bin.
    function weighted_median_age(pop::AbstractVector{<:Real})
        half = sum(pop) / 2
        k = searchsortedfirst(cumsum(pop), half)
        before = k == 1 ? 0.0 : sum(@view pop[1:k-1])
        return (k - 1) + (half - before) / pop[k]
    end

    summary = DataFrame(
        year = Int[], total_population = Float64[], median_age = Float64[],
        share_65_plus_pct = Float64[],
    )
    by_age = DataFrame(
        year = Int[], age = Int[], population = Float64[], share_pct = Float64[],
    )
    for year in years
        pop = populations[year]
        total = sum(pop)
        push!(summary, (
            year, total, weighted_median_age(pop),
            100 * sum(@view pop[66:end]) / total,
        ))
        for (idx, count) in enumerate(pop)
            push!(by_age, (year, idx - 1, count, 100 * count / total))
        end
    end

    CSV.write(joinpath(output_dir, "population_age_summary.csv"), summary)
    CSV.write(joinpath(output_dir, "population_by_single_year_age.csv"), by_age)

    # Step-outline coordinates for each 1-year age bin; end at zero to close
    # the lightly filled polygon. This produces the Manhattan-like silhouette.
    function step_outline(values::AbstractVector{<:Real})
        x = Float64[0.0]
        y = Float64[0.0]
        for (i, value) in enumerate(values)
            push!(x, i - 1.0, Float64(i))
            push!(y, Float64(value), Float64(value))
        end
        push!(x, Float64(length(values)))
        push!(y, 0.0)
        return x, y
    end

    colors = [:royalblue, :darkorange, :seagreen]

    fig = CairoMakie.Figure(size = (1160, 820), fontsize = 16)
    CairoMakie.Label(
        fig[0, 1],
        "Kiribati population projections: 2030, 2040 and 2050";
        fontsize = 22, font = :bold, halign = :left,
    )
    ax_share = CairoMakie.Axis(
        fig[1, 1];
        title = "Age composition (population size normalised)",
        ylabel = "Share of population at each age (%)",
        xticks = 0:10:top_age,
        xgridvisible = false,
        ygridvisible = true,
    )
    ax_count = CairoMakie.Axis(
        fig[2, 1];
        title = "Age-specific population counts",
        xlabel = "Age (years)",
        ylabel = "People at each age",
        xticks = 0:10:top_age,
        xgridvisible = false,
        ygridvisible = true,
    )

    # Draw all translucent fills first and then all outlines, to prevent
    # one year's fill from hiding another year's edge.
    for (year, color) in zip(years, colors)
        pop = populations[year]
        values_share = 100 .* pop ./ sum(pop)
        for (ax, values) in ((ax_share, values_share), (ax_count, pop))
            xs, ys = step_outline(values)
            points = [CairoMakie.Point2f(x, y) for (x, y) in zip(xs, ys)]
            CairoMakie.poly!(ax, points; color = (color, 0.09), strokewidth = 0)
        end
    end

    for (year, color) in zip(years, colors)
        pop = populations[year]
        stats = only(eachrow(summary[summary.year .== year, :]))
        legend_label = @sprintf(
            "%d   median %.1f y   age 65+: %.1f%%", year,
            stats.median_age, stats.share_65_plus_pct,
        )
        xs, ys = step_outline(100 .* pop ./ sum(pop))
        CairoMakie.lines!(ax_share, xs, ys;
            color = color, linewidth = 2.8, label = legend_label)
        xs, ys = step_outline(pop)
        CairoMakie.lines!(ax_count, xs, ys;
            color = color, linewidth = 2.8)
    end

    # 65+ boundary as a shared visual reference, without curve annotations.
    for ax in (ax_share, ax_count)
        CairoMakie.vlines!(ax, [65.0]; color = (:gray45, 0.6),
            linewidth = 1.4, linestyle = :dash)
        CairoMakie.xlims!(ax, 0, n_ages)
    end
    CairoMakie.ylims!(ax_share, 0, 1.08 * maximum(
        maximum(100 .* populations[y] ./ sum(populations[y])) for y in years))
    CairoMakie.ylims!(ax_count, 0, 1.08 * maximum(
        maximum(populations[y]) for y in years))

    CairoMakie.axislegend(ax_share; position = :rt,
        framevisible = false, labelsize = 13)
    CairoMakie.Label(fig[3, 1],
        "Dashed line: age 65. Last age category in get_population() is shown as $(top_age)+ if top-coded.";
        fontsize = 12, color = :gray40, halign = :left)
    CairoMakie.rowgap!(fig.layout, 15)

    CairoMakie.save(joinpath(output_dir,
        "population_age_distributions_2030_2040_2050.png"), fig; px_per_unit = 2)
    CairoMakie.save(joinpath(output_dir,
        "population_age_distributions_2030_2040_2050.svg"), fig)

    println("\nKiribati population age distribution comparison")
    @printf("%-8s %18s %15s %15s\n", "Year", "Total population", "Median age", "65+ share (%)")
    for row in eachrow(summary)
        @printf("%-8d %18.0f %15.1f %15.2f\n",
            row.year, row.total_population, row.median_age, row.share_65_plus_pct)
    end
    println("\nFigures and CSV tables saved to: $output_dir")
    display(fig)
    fig
end
