using CairoMakie

"""
    plot_contact_matrix(contact; kwargs...)

Plot an age-specific contact matrix.
Rows represent recipient ages and columns
represent source ages.
"""
function plot_contact_matrix(
    contact::AbstractMatrix;
    title="Age-specific mixing",
    colorlabel="Contacts",
    colorrange=nothing,
    colormap=:viridis,
)
    validate_contact_matrix(contact)

    ages = collect(0:(size(contact, 1)-1))

    fig = Figure(size=(600, 520))

    ax = Axis(
        fig[1, 1];
        title,
        xlabel="Contact age",
        ylabel="Participant age",
        aspect=DataAspect(),
    )

    limits = isnothing(colorrange) ?
             (0.0, maximum(contact)) : colorrange

    hm = heatmap!(
        ax, ages, ages, permutedims(contact);
        colormap,
        colorrange=limits,
    )

    Colorbar(fig[1, 2], hm; label=colorlabel)

    return fig
end


"""
    plot_contact_intensity(contact; kwargs...)

Plot total contact intensity by participant age,
optionally accounting for source infectiousness.
"""
function plot_contact_intensity(
    contact::AbstractMatrix;
    infectiousness=nothing,
    ylabel="Contact intensity",
)
    validate_contact_matrix(contact)

    ages = collect(0:(size(contact, 1)-1))

    fig = Figure(size=(750, 400))

    ax = Axis(
        fig[1, 1];
        xlabel="Age",
        ylabel,
    )

    lines!(
        ax, ages, vec(sum(contact; dims=2));
        linewidth=2.5,
        label="All contacts",
    )

    if !isnothing(infectiousness)
        lines!(
            ax, ages, contact * infectiousness;
            linewidth=2.5,
            label="Potentially infectious contacts",
        )

        axislegend(ax; position=:rt)
    end

    return fig
end