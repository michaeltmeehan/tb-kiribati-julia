
using CSV
using DataFrames
using Statistics
using LinearAlgebra: norm

let
    root = normpath(joinpath(@__DIR__, "..", ".."))

    include(joinpath(
        root, "src", "mixing", "prem_reciprocity.jl"
    ))

    constmod = PremReciprocity

    prem = constmod.PremContacts(
        joinpath(
            root, "data", "raw", "prem",
            "solomon_islands_all_16x16.csv"
        ),
        joinpath(
            root, "data", "processed", "demography",
            "population.csv"
        ),
    )

    outdir = joinpath(
        root, "data", "processed", "contacts"
    )
    mkpath(outdir)

    # --------------------------------------------------
    # Fixed 2023 correction
    # --------------------------------------------------

    C16 = constmod.source_matrix(prem, 2023)
    C96 = constmod.contact_matrix(prem, 2023)

    N = constmod.population_at(prem, 2023)

    err = constmod.reciprocity_error(C16, N)
    scale = maximum(N .* C16)

    @assert err <= 1e-12 * scale

    # The 96-age expansion preserves source-band totals.
    groups = [min(div(a, 5) + 1, 16) for a in 0:95]

    for i in 1:16, j in 1:16
        a = findfirst(==(i), groups)
        b = findall(==(j), groups)

        @assert isapprox(
            sum(C96[a, b]), C16[i, j];
            atol=1e-12
        )
    end

    # --------------------------------------------------
    # Save fixed contact matrices
    # --------------------------------------------------

    CSV.write(
        joinpath(outdir, "prem_reciprocal_2023_16.csv"),
        DataFrame(C16, :auto);
        writeheader=false
    )

    output = DataFrame(recipient_age=collect(0:95))

    for age in 0:95
        output[!, Symbol("source_age_$age")] =
            C96[:, age+1]
    end

    CSV.write(
        joinpath(outdir, "prem_reciprocal_2023_96.csv"),
        output
    )

    # --------------------------------------------------
    # Verify time-varying correction
    # --------------------------------------------------

    println("\nRECIPROCITY VALIDATION")

    for year in (1950, 2000, 2023, 2050, 2100)
        C = constmod.source_matrix(prem, year)
        P = constmod.population_at(prem, year)

        error = constmod.reciprocity_error(C, P)
        relative_error = error / maximum(P .* C)

        @assert relative_error < 1e-12

        println(
            "Year ", year,
            ": relative reciprocity error = ",
            relative_error
        )
    end

    # Compare original and adjusted contact patterns.
    original = constmod.contact_matrix(
        prem, 2023; reciprocal=false
    )

    println("\nFIXED 2023 COMPARISON")
    println("Original mean row sum: ",
        mean(sum(original, dims=2)))

    println("Adjusted mean row sum: ",
        mean(sum(C96, dims=2)))

    println("Maximum absolute change: ",
        maximum(abs.(C96 .- original)))

    println("Relative Frobenius change: ",
        norm(C96 .- original) / norm(original))

    println("\nSaved to: ", outdir)
end
