
using CSV
using DataFrames
using DelimitedFiles
using Statistics

let
    root = normpath(joinpath(@__DIR__, "..", ".."))

    rawdir = joinpath(root, "data", "raw", "prem")
    outdir = joinpath(root, "data", "processed", "contacts")
    mkpath(outdir)

    rawfile = joinpath(
        rawdir, "solomon_islands_all_16x16.csv"
    )

    source = Matrix{Float64}(readdlm(rawfile, ','))

    @assert size(source) == (16, 16)
    @assert all(isfinite, source)
    @assert all(>=(0), source)

    # -------------------------------------------------------
    # Expand 16 age groups to 96 single-year ages
    # -------------------------------------------------------

    ages = collect(0:95)

    # Original groups:
    # 0-4, 5-9, ..., 70-74, 75+
    group = [min(fld(a, 5) + 1, 16) for a in ages]

    group_sizes = [
        count(==(g), group) for g in 1:16
    ]

    @assert group_sizes == [fill(5, 15); 21]

    contact = [
        source[group[a], group[b]] /
            group_sizes[group[b]]
        for a in 1:96, b in 1:96
    ]

    # -------------------------------------------------------
    # Validate conservation of source-band contacts
    # -------------------------------------------------------

    reconstructed = [
        sum(contact[
            findfirst(==(i), group),
            findall(==(j), group)
        ])
        for i in 1:16, j in 1:16
    ]

    conservation_error =
        maximum(abs.(reconstructed .- source))

    @assert conservation_error < 1e-12

    # -------------------------------------------------------
    # Compare with legacy model input
    # -------------------------------------------------------

    legacy = Matrix{Float64}(
        readdlm(joinpath(root, "data", "contact"))
    )

    @assert size(legacy) == size(contact)

    difference = abs.(legacy .- contact)

    println("\nCONTACT MATRIX VALIDATION")
    println("Source dimensions: ", size(source))
    println("Expanded dimensions: ", size(contact))
    println("Conservation error: ", conservation_error)
    println("Maximum legacy difference: ", maximum(difference))
    println("Mean legacy difference: ", mean(difference))

    # Allow for rounding in the legacy text file.
    @assert maximum(difference) < 1e-5 """
        Contact reconstruction differs from legacy data.
        Check source version, orientation and transformation.
    """

    # -------------------------------------------------------
    # Save self-describing processed matrix
    # -------------------------------------------------------

    output = DataFrame(recipient_age=ages)

    for (j, age) in enumerate(ages)
        output[!, Symbol("source_age_$age")] =
            contact[:, j]
    end

    CSV.write(
        joinpath(outdir, "contact_matrix.csv"),
        output
    )

    # -------------------------------------------------------
    # Document processing assumptions
    # -------------------------------------------------------

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
        ]
    )

    CSV.write(
        joinpath(outdir, "metadata.csv"),
        metadata
    )

    println("\nReconstruction successful.")
end
