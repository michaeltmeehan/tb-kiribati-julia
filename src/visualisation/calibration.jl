using CairoMakie

"""Plot predicted and observed national notification totals."""
function plot_calibration_national(
    fit::AbstractCalibrationFit;
    diagnostics=calibration_diagnostics(fit),
)
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


"""Age-group notification fit, including combined age groups."""
function plot_calibration_by_age(
    fit::AbstractCalibrationFit;
    diagnostics=calibration_diagnostics(fit),
)
    labels = unique([
        r.age_group for r in diagnostics.age
                        if r.age_group != "all"
    ])

    fig = Figure(size=(1100, 350 * cld(length(labels), 3)))

    for (k, label) in enumerate(labels)
        # Select and sort observations for this age group.
        group = [
            r for r in diagnostics.age
                  if r.age_group == label
        ]
        sort!(group, by=r -> r.year)

        group_years = [r.year for r in group]

        ax = Axis(
            fig[cld(k, 3), mod1(k, 3)],
            title=label,
            xlabel="Year",
            ylabel="Notifications",
        )

        lines!(
            ax,
            group_years,
            [r.expected for r in group],
            color=:steelblue,
            linewidth=2,
        )

        scatter!(
            ax,
            group_years,
            [r.observed for r in group],
            color=:black,
        )

        xlims!(
            ax,
            minimum(group_years) - 0.5,
            maximum(group_years) + 0.5,
        )
    end

    return fig
end

