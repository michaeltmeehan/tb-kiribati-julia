using CairoMakie

"""Plot predicted and observed national notification totals."""
function plot_calibration_national(fit::MAPFit; diagnostics=calibration_diagnostics(fit))
    fig = Figure(size=(850, 480))
    ax = Axis(fig[1, 1], xlabel="Year", ylabel="Annual notifications",
        title="MAP calibration: national notifications")
    rows = diagnostics.national
    lines!(ax, [r.year for r in rows], [r.expected for r in rows],
        color=:steelblue, linewidth=2, label="Model")
    scatter!(ax, [r.year for r in rows], [r.observed for r in rows],
        color=:black, label="Observed")
    axislegend(ax, position=:lt)
    return fig
end

"""Age-group notification fit, respecting 2024's combined childhood observation."""
function plot_calibration_by_age(fit::MAPFit; diagnostics=calibration_diagnostics(fit))
    # One panel per observed label, including any aggregate such as 0-14.
    labels = unique([r.age_group for r in diagnostics.age])
    fig = Figure(size=(1100, 350*cld(length(labels), 3)))
    for (k, label) in enumerate(labels)
        ax = Axis(fig[cld(k, 3), mod1(k, 3)], title=label,
            xlabel="Year", ylabel="Notifications")
        group = [r for r in diagnostics.age if r.age_group==label]
        sort!(group, by=r -> r.year)
        lines!(ax, [r.year for r in group], [r.expected for r in group],
            color=:steelblue, linewidth=2)
        scatter!(ax, [r.year for r in group], [r.observed for r in group], color=:black)
    end
    return fig
end
