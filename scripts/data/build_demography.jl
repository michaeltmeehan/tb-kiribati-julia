
using CSV
using DataFrames

let
    root = normpath(joinpath(@__DIR__, "..", ".."))

    include(joinpath(
        root, "src", "data", "wpp2024_processing.jl"
    ))

    rawdir = joinpath(root, "data", "raw", "wpp2024")
    outdir = joinpath(root, "data", "processed", "demography")

    readraw(name) = CSV.read(
        joinpath(rawdir, "$name.csv"), DataFrame
    )

    result = WPP2024Processing.process_demography(
        readraw("population"),
        readraw("fertility"),
        readraw("mortality"),
        readraw("migration"),
        readraw("vital_totals"),
    )

    mkpath(outdir)

    for (name, df) in pairs(result)
        CSV.write(joinpath(outdir, "$name.csv"), df)
        println(rpad(string(name), 22), nrow(df), " rows")
    end

    println("\nSelected population totals:")
    println(filter(
        :year => in([1950, 2000, 2023, 2025, 2050, 2100]),
        result.population_totals
    ))

    println("\nMaximum TFR reconstruction error: ",
        maximum(abs.(
            result.fertility_checks.tfr_reconstructed .-
                result.fertility_checks.tfr_reported
        ))
    )
end
