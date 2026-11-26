using CairoMakie
using DataFrames
using Statistics

"""
    predictive_intervals(draws, metric; group=:year)

Calculate pointwise median, 50% and 95% predictive
intervals from simulation draws.
"""
function predictive_intervals(
    draws::AbstractDataFrame,
    metric::Symbol;
    group::Symbol=:year,
)
    rows = NamedTuple[]

    for g in groupby(draws, group; sort=true)
        values = filter(
            isfinite,
            Float64.(collect(skipmissing(g[!, metric])))
        )

        isempty(values) && continue

        q = quantile(
            values, [0.025, 0.25, 0.5, 0.75, 0.975]
        )

        push!(rows, (
            group=g[1, group],
            n=length(values),
            q025=q[1],
            q25=q[2],
            median=q[3],
            q75=q[4],
            q975=q[5],
        ))
    end

    return DataFrame(rows)
end


"""
    plot_predictive!(ax, summary; kwargs...)

Draw predictive intervals on an existing Makie axis.
"""
function plot_predictive!(
    ax,
    summary::AbstractDataFrame;
    color=:steelblue,
    label="Predictive median",
    scale=1.0,
)
    x = summary.group

    band!(
        ax, x,
        scale .* summary.q025,
        scale .* summary.q975;
        color=(color, 0.15),
        label="95% interval",
    )

    band!(
        ax, x,
        scale .* summary.q25,
        scale .* summary.q75;
        color=(color, 0.30),
        label="50% interval",
    )

    lines!(
        ax, x,
        scale .* summary.median;
        color,
        linewidth=2,
        label,
    )

    return ax
end


"""
    plot_predictive_summary(draws, metrics; kwargs...)

Generate a multi-panel predictive summary figure.

Optional observations are supplied as a dictionary
mapping metrics to tables with year and value columns.
"""
function plot_predictive_summary(
    draws::AbstractDataFrame,
    metrics;
    labels=Dict(),
    scales=Dict(),
    observations=Dict(),
    ncols=2,
    color=:steelblue,
    prediction_label="Predictive median",
    log_metrics=Symbol[],
    year_step=2,
)
    nrows = cld(length(metrics), ncols)

    fig = Figure(
        size=(550 * ncols, 320 * nrows)
    )

    for (i, metric) in enumerate(metrics)

        summary = predictive_intervals(draws, metric)
        scale = get(scales, metric, 1.0)

        use_log = metric in log_metrics &&
            all(>(0), summary.q025)

        if use_log && haskey(observations, metric)
            obs = observations[metric]

            use_log = all(>(0), obs.value)

            if :lower in propertynames(obs)
                use_log = use_log && all(>(0), obs.lower)
            end
        end

        if metric in log_metrics && !use_log
            @warn "Using linear scale for $metric due to non-positive values"
        end

        ax = Axis(
            fig[cld(i, ncols), mod1(i, ncols)];
            title=get(labels, metric, string(metric)),
            xlabel="Year",
            yscale=use_log ? log10 : identity,
            xticks=minimum(draws.year):year_step:maximum(draws.year),
        )

        plot_predictive!(
            ax, summary;
            color,
            label=prediction_label,
            scale,
        )

        if haskey(observations, metric)
            obs = observations[metric]

            is_who = :source in propertynames(obs) &&
                all(==("WHO estimate"), obs.source)

            obs_color = is_who ? :darkorange : :black
            obs_label = is_who ? "WHO estimate" : "Observed"
            obs_marker = is_who ? :diamond : :circle

            if all(
                c -> c in propertynames(obs),
                (:lower, :upper)
            )
                rangebars!(
                    ax,
                    obs.year,
                    scale .* obs.lower,
                    scale .* obs.upper;
                    color=obs_color,
                    whiskerwidth=6,
                    linewidth=1.5,
                )
            end

            scatter!(
                ax,
                obs.year,
                scale .* obs.value;
                color=obs_color,
                marker=obs_marker,
                markersize=7,
                label=obs_label,
            )
        end

        if !use_log
            ylims!(ax, 0, nothing)
        end

        axislegend(
            ax;
            position=:rt,
            labelsize=10,
        )
    end

    return fig
end

"""
    plot_predictive_comparison(first, second, metrics; kwargs...)

Compare two sets of predictive draws using their
median and pointwise 95% intervals.
"""
function plot_predictive_comparison(
    first_draws::AbstractDataFrame,
    second_draws::AbstractDataFrame,
    metrics;
    labels=Dict(),
    scales=Dict(),
    observations=Dict(),
    series_labels=("Prior", "Posterior"),
    colors=(:grey50, :steelblue),
    ncols=2,
)
    nrows = cld(length(metrics), ncols)
    fig = Figure(size=(550*ncols, 320*nrows))

    for (i, metric) in enumerate(metrics)
        ax = Axis(
            fig[cld(i, ncols), mod1(i, ncols)];
            title=get(labels, metric, string(metric)),
            xlabel="Year",
        )

        scale = get(scales, metric, 1.0)

        for (draws, color, label) in zip(
            (first_draws, second_draws),
            colors,
            series_labels,
        )
            s = predictive_intervals(draws, metric)

            band!(
                ax, s.group,
                scale .* s.q025,
                scale .* s.q975;
                color=(color, 0.2),
            )

            lines!(
                ax, s.group,
                scale .* s.median;
                color,
                linewidth=2,
                label,
            )
        end

        if haskey(observations, metric)
            obs = observations[metric]

            is_who = :source in propertynames(obs) &&
                all(==("WHO estimate"), obs.source)

            obs_color = is_who ? :darkorange : :black
            obs_label = is_who ? "WHO estimate" : "Observed"
            obs_marker = is_who ? :diamond : :circle

            if all(
                c -> c in propertynames(obs),
                (:lower, :upper)
            )
                rangebars!(
                    ax,
                    obs.year,
                    scale .* obs.lower,
                    scale .* obs.upper;
                    color=obs_color,
                    whiskerwidth=6,
                    linewidth=1.5,
                )
            end

            scatter!(
                ax,
                obs.year,
                scale .* obs.value;
                color=obs_color,
                marker=obs_marker,
                markersize=7,
                label=obs_label,
            )
        end

        ylims!(ax, 0, nothing)
        axislegend(ax; position=:rt, labelsize=10)
    end

    return fig
end

"""
    plot_predictive_by_age(draws, metrics; kwargs...)

Plot age-specific predictive medians and
50% and 95% intervals.
"""
function plot_predictive_by_age(
    draws::AbstractDataFrame,
    metrics;
    age_labels,
    labels=Dict(),
    observations=Dict(),
    color=:steelblue,
    prediction_label="Predictive median",
)
    fig = Figure(size=(550*length(metrics), 450))

    for (i, metric) in enumerate(metrics)
        ax = Axis(
            fig[1, i];
            title=get(labels, metric, string(metric)),
            xlabel="Age group",
            xticks=(1:length(age_labels), age_labels),
            xticklabelrotation=π/4,
        )

        s = predictive_intervals(
            draws, metric;
            group=:age_group,
        )

        x = [
            findfirst(==(g), age_labels)
            for g in s.group
        ]

        all(v -> !isnothing(v), x) ||
            error("Unrecognised age group")

        x = Int.(x)

        rangebars!(
            ax, x, s.q025, s.q975;
            color,
            linewidth=1.5,
            whiskerwidth=8,
            label="95% interval",
        )

        rangebars!(
            ax, x, s.q25, s.q75;
            color,
            linewidth=5,
            label="50% interval",
        )

        scatter!(
            ax, x, s.median;
            color,
            markersize=9,
            label=prediction_label,
        )

        if haskey(observations, metric)
            obs = observations[metric]

            ox = Int[
                findfirst(==(g), age_labels)
                for g in obs.age_group
            ]

            scatter!(
                ax, ox, obs.value;
                color=:black,
                marker=:diamond,
                markersize=10,
                label="Observed",
            )
        end

        ylims!(ax, 0, nothing)
        axislegend(ax; position=:rt, labelsize=10)
    end

    return fig
end

"""
    plot_predictive_groups(draws; kwargs...)

Plot expected and replicated observations across
multiple groups, with one time-series panel per group.

Draws must contain year, expected, predictive,
observed and the specified group column.
"""
function plot_predictive_groups(
    draws::AbstractDataFrame;
    group=:age_group,
    group_order=unique(draws[!, group]),
    labels=Dict(),
    ncols=3,
    scale=1.0,
    ylabel="",
    year_step=3,
)
    nrows = cld(length(group_order), ncols)

    fig = Figure(
        size=(420*ncols, 300*nrows + 60)
    )

    for (i, name) in enumerate(group_order)

        d = filter(
            r -> r[group] == name,
            draws,
        )

        isempty(d) && continue

        ax = Axis(
            fig[cld(i, ncols), mod1(i, ncols)];
            title=get(labels, name, string(name)),
            xlabel="Year",
            ylabel,
            xticks=minimum(d.year):year_step:maximum(d.year),
        )

        expected = predictive_intervals(d, :expected)
        predictive = predictive_intervals(d, :predictive)

        # Full observation uncertainty
        band!(
            ax,
            predictive.group,
            scale .* predictive.q025,
            scale .* predictive.q975;
            color=(:steelblue, 0.15),
            label="Predictive 95%",
        )

        # Parameter uncertainty
        band!(
            ax,
            expected.group,
            scale .* expected.q025,
            scale .* expected.q975;
            color=(:steelblue, 0.35),
            label="Parameter 95%",
        )

        lines!(
            ax,
            expected.group,
            scale .* expected.median;
            color=:steelblue,
            linewidth=2,
            label="Posterior median",
        )

        obs = unique(
            select(d, :year, :observed)
        )

        dropmissing!(obs)

        scatter!(
            ax,
            obs.year,
            scale .* obs.observed;
            color=:black,
            markersize=7,
            label="Observed",
        )

        ylims!(ax, 0, nothing)

        if i == 1
            Legend(
                fig[nrows+1, 1:ncols],
                ax;
                orientation=:horizontal,
                framevisible=false,
            )
        end
    end

    return fig
end