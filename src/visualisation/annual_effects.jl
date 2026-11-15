
"""
    plot_annual_effects(effects)

Visualise conditional posterior annual multipliers
and national notification counts.

`effects` is returned by posterior_annual_effects.
"""
function plot_annual_effects(effects)
    years = [r.year for r in effects]

    means = [r.mean_multiplier for r in effects]
    lower = [r.lower_multiplier for r in effects]
    upper = [r.upper_multiplier for r in effects]

    observed = [r.observed for r in effects]
    expected = [r.expected for r in effects]
    conditional = [
        r.conditional_expected for r in effects
    ]

    fig = Figure(size=(950, 750))

    ax1 = Axis(
        fig[1, 1];
        xlabel="Year",
        ylabel="Annual notification multiplier",
        title="Conditional posterior annual effects",
    )

    hlines!(
        ax1,
        [1.0];
        color=:gray,
        linestyle=:dash,
    )

    rangebars!(
        ax1,
        years,
        lower,
        upper;
        color=:steelblue,
    )

    scatter!(
        ax1,
        years,
        means;
        color=:steelblue,
        markersize=10,
    )

    ax2 = Axis(
        fig[2, 1];
        xlabel="Year",
        ylabel="Annual notifications",
        title="Marginal and conditional fitted counts",
    )

    lines!(
        ax2,
        years,
        expected;
        color=:steelblue,
        linewidth=2,
        label="Marginal expectation",
    )

    lines!(
        ax2,
        years,
        conditional;
        color=:darkorange,
        linewidth=2,
        label="Conditional expectation",
    )

    scatter!(
        ax2,
        years,
        observed;
        color=:black,
        markersize=10,
        label="Observed",
    )

    axislegend(ax2; position=:lt)

    return fig
end
