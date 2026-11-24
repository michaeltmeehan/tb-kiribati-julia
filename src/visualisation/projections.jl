using CairoMakie
using DataFrames

const PROJECTION_SCENARIOS = (
    :fixed,
    :growth_only,
    :projected,
)

const PROJECTION_LABELS = (
    fixed="Fixed population",
    growth_only="Population growth only",
    projected="Full demographic projection",
)

const PROJECTION_COLOURS = (
    fixed=:gray,
    growth_only=:steelblue,
    projected=:darkorange,
)

const PROJECTION_LINESTYLES = (
    fixed=:dash,
    growth_only=:dot,
    projected=:solid,
)

const PROJECTION_YLABELS = (
    population="Population",
    proportion_65_plus="Population aged 65+ (%)",
    incidence="Annual TB cases",
    notifications="Annual TB notifications",
    mortality="Annual TB deaths",
    incidence_rate="Incidence per 100,000",
    notification_rate="Notifications per 100,000",
    mortality_rate="Mortality per 100,000",
    latent_prevalence="Latent TB prevalence (%)",
    active_prevalence="Active TB prevalence per 100,000",
)

const PROJECTION_EFFECTS = (
    :population_growth,
    :demographic_composition,
    :combined_demography,
)

const PROJECTION_EFFECT_LABELS = (
    population_growth="Population growth",
    demographic_composition="Composition and turnover",
    combined_demography="Combined demography",
)

const PROJECTION_EFFECT_COLOURS = (
    population_growth=:steelblue,
    demographic_composition=:darkorange,
    combined_demography=:black,
)


"""
    plot_projections!(ax, national, variable; scenarios)

Plot deterministic scenario trajectories on an existing axis.
"""
function plot_projections!(
    ax::Axis,
    national::AbstractDataFrame,
    variable::Symbol;
    scenarios=PROJECTION_SCENARIOS,
)
    scale = variable in (
        :proportion_65_plus,
        :latent_prevalence,
    ) ? 100.0 : 1.0

    ax.xlabel = "Year"
    ax.ylabel = get(
        PROJECTION_YLABELS, variable, string(variable)
    )

    for scenario in scenarios

        df = sort(
            filter(r -> r.scenario == scenario, national),
            :year,
        )

        isempty(df) && continue

        allunique(df.year) ||
            throw(ArgumentError(
                "Multiple draws detected. Summarise ensemble first."
            ))

        lines!(
            ax,
            df.year,
            scale .* df[!, variable];
            color=PROJECTION_COLOURS[scenario],
            linestyle=PROJECTION_LINESTYLES[scenario],
            label=PROJECTION_LABELS[scenario],
            linewidth=2,
        )
    end

    return ax
end


"""
    plot_projections(national; variables, ncols, scenarios)

Plot national demographic and epidemiological projections.
"""
function plot_projections(
    national::AbstractDataFrame;
    variables=(
        :incidence,
        :incidence_rate,
        :notifications,
        :notification_rate,
        :mortality,
        :mortality_rate,
    ),
    ncols::Int=2,
    scenarios=PROJECTION_SCENARIOS,
)
    nrows = cld(length(variables), ncols)

    fig = Figure(
        size=(550 * ncols, 320 * nrows + 80)
    )

    axes = Axis[]

    for (k, variable) in enumerate(variables)

        ax = Axis(
            fig[cld(k, ncols), mod1(k, ncols)]
        )

        plot_projections!(
            ax,
            national,
            variable;
            scenarios,
        )

        push!(axes, ax)
    end

    linkxaxes!(axes...)

    Legend(
        fig[nrows+1, 1:ncols],
        first(axes);
        orientation=:horizontal,
        framevisible=false,
    )

    return fig
end


"""
    plot_projection_effects!(ax, comparison, variable; measure)

Plot paired demographic contrasts on an existing axis.
"""
function plot_projection_effects!(
    ax::Axis,
    comparison::AbstractDataFrame,
    variable::Symbol;
    measure::Symbol=:percentage_change,
)
    labels = (
        difference="Absolute difference",
        ratio="Ratio",
        percentage_change="Percentage change (%)",
    )

    haskey(labels, measure) ||
        throw(ArgumentError("Unsupported comparison measure"))

    ax.xlabel = "Year"
    ax.ylabel = labels[measure]
    ax.title = get(
        PROJECTION_YLABELS, variable, string(variable)
    )

    hlines!(
        ax,
        [measure == :ratio ? 1.0 : 0.0];
        color=:gray,
        linestyle=:dash,
    )

    for effect in PROJECTION_EFFECTS

        df = sort(
            filter(
                r -> r.variable == variable &&
                    r.effect == effect,
                comparison,
            ),
            :year,
        )

        isempty(df) && continue

        allunique(df.year) ||
            throw(ArgumentError(
                "Multiple draws detected. Summarise ensemble first."
            ))

        values = Float64.(coalesce.(df[!, measure], NaN))

        lines!(
            ax,
            df.year,
            values;
            color=PROJECTION_EFFECT_COLOURS[effect],
            label=PROJECTION_EFFECT_LABELS[effect],
            linewidth=2,
        )
    end

    return ax
end


"""
    plot_projection_effects(comparison; variables, measure, ncols)

Plot the decomposition of demographic effects.
"""
function plot_projection_effects(
    comparison::AbstractDataFrame;
    variables=(
        :incidence,
        :incidence_rate,
        :mortality,
        :mortality_rate,
    ),
    measure::Symbol=:percentage_change,
    ncols::Int=2,
)
    nrows = cld(length(variables), ncols)

    fig = Figure(
        size=(550 * ncols, 320 * nrows + 80)
    )

    axes = Axis[]

    for (k, variable) in enumerate(variables)

        ax = Axis(
            fig[cld(k, ncols), mod1(k, ncols)]
        )

        plot_projection_effects!(
            ax,
            comparison,
            variable;
            measure,
        )

        push!(axes, ax)
    end

    linkxaxes!(axes...)

    Legend(
        fig[nrows+1, 1:ncols],
        first(axes);
        orientation=:horizontal,
        framevisible=false,
    )

    return fig
end