
using CSV
using DataFrames
using DelimitedFiles
using Statistics

let
    root = normpath(joinpath(@__DIR__, "..", ".."))

    include(joinpath(
        root, "src", "data", "prem_processing.jl"
    ))

    rawdir = joinpath(root, "data", "raw", "prem")
    outdir = joinpath(root, "data", "processed", "contacts")
    mkpath(outdir)

    source = Matrix{Float64}(readdlm(
        joinpath(rawdir, "solomon_islands_all_16x16.csv"),
        ','
    ))

    expanded = PremProcessing.expand_contacts(source)
    recovered = PremProcessing.collapse_contacts(expanded)

    conservation_error = maximum(abs.(recovered .- source))

    @assert conservation_error < 1e-12

    # Compare against the original model input.
    legacy = Matrix{Float64}(readdlm(
        joinpath(root, "data", "contact"),
        Float64
    ))

    @assert size(legacy) == size(expanded)

    difference = abs.(legacy .- expanded)

    println("\nCONTACT MATRIX VALIDATION")
    println("Conservation error: ", conservation_error)
    println("Maximum legacy difference: ", maximum(difference))
    println("Mean legacy difference: ", mean(difference))

    @assert maximum(difference) < 1e-5

    CSV.write(
        joinpath(outdir, "contact_matrix.csv"),
        PremProcessing.contact_dataframe(expanded)
    )

    metadata = DataFrame(
        key=[
            "source_country",
            "target_country",
            "source_age_groups",
            "target_ages",
            "orientation",
            "adaptation",
            "source_age_allocation",
            "population_weighting",
            "reciprocity_adjustment",
            "normalisation",
        ],
        value=[
            "Solomon Islands",
            "Kiribati",
            "0-4,...,70-74,75+",
            "0-95",
            "recipient_source",
            "source_band",
            "equal_within_band",
            "none",
            "none",
            "none_in_data_processing",
        ],
    )

    CSV.write(joinpath(outdir, "metadata.csv"), metadata)

    println("Contact reconstruction successful.")
end
