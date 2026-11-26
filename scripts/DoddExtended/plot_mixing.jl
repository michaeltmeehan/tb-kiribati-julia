using TBKiribatiJulia
using CairoMakie

include(joinpath(
    @__DIR__, "..", "common", "kiribati_paths.jl"
))

let
    TB = TBKiribatiJulia

    root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = kiribati_data_paths(root)

    output_dir = joinpath(root, "results", "mixing")
    mkpath(output_dir)

    # Load mixing inputs
    contact = TB.load_contact_matrix(
        paths.legacy_contact;
        nages=96,
    )

    C = TB.normalise_model_contacts(contact)
    w = TB.adult_infectiousness(size(C, 1))

    # Effective transmission mixing
    C_eff = C .* reshape(w, 1, :)

    figures = (
        raw_contacts=TB.plot_contact_matrix(
            contact;
            title="Age-specific social contacts",
            colorlabel="Contacts per day",
        ),
        normalised_contacts=TB.plot_contact_matrix(
            C;
            title="Normalised contact matrix",
            colorlabel="Relative contact intensity",
        ),
        effective_mixing=TB.plot_contact_matrix(
            C_eff;
            title="Effective transmission mixing",
            colorlabel="Relative contact intensity",
            colorrange=(0.0, maximum(C)),
        ),
        contact_intensity=TB.plot_contact_intensity(
            C;
            infectiousness=w,
            ylabel="Relative contact intensity",
        ),
    )

    for (name, fig) in pairs(figures)
        display(fig)

        save(
            joinpath(output_dir, "$name.png"),
            fig;
            px_per_unit=3,
        )

        save(joinpath(output_dir, "$name.svg"), fig)
    end
end