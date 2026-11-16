
using CairoMakie

"""
    plot_profile_likelihood(profile; threshold=3.841459)

Plot a one-dimensional profile deviance.
"""
function plot_profile_likelihood(
    profile;
    threshold=3.841459,
)
    points = profile.points

    x = [p.value for p in points]
    y = [p.deviance for p in points]

    fig = Figure(size=(750, 450))

    ax = Axis(
        fig[1, 1];
        xlabel=string(profile.parameter),
        ylabel="Profile deviance",
        title="Profile likelihood: $(profile.parameter)",
    )

    lines!(ax, x, y; linewidth=2)
    scatter!(ax, x, y; markersize=7)

    hlines!(
        ax,
        [threshold];
        color=:gray,
        linestyle=:dash,
    )

    vlines!(
        ax,
        [profile.mle];
        color=:black,
        linestyle=:dot,
    )

    return fig
end
