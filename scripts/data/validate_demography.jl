
using CSV
using DataFrames
using DelimitedFiles
using Statistics

# Repository root (relative to this script)
root = normpath(joinpath(@__DIR__, "..", ".."))

olddir = joinpath(root, "data")
newdir = joinpath(olddir, "processed", "demography")

# ---------------------------------------------------------
# Compare population, fertility and mortality
# ---------------------------------------------------------

specifications = [
    ("population", :pop, 0:100, 1 / 1000),
    ("fertility", :asfr, 10:54, 1.0),
    ("mortality", :mxB, 0:100, 1.0),
]

println("\nDEMOGRAPHIC INPUT VALIDATION")
println("-"^65)

for (name, field, ages, conversion) in specifications

    new = CSV.read(
        joinpath(newdir, "$name.csv"),
        DataFrame,
    )

    old = readdlm(
        joinpath(olddir, name),
        Float64,
    )

    # Reconstruct the legacy matrix layout:
    # rows = ages, columns = years (1950–2100).
    rebuilt = [
        only(new[
            (new.year .== y) .& (new.age .== a),
            field,
        ]) * conversion
        for a in ages, y in 1950:2100
    ]

    @assert size(old) == size(rebuilt) """
        Dimension mismatch for $name:
        old = $(size(old)), new = $(size(rebuilt))
    """

    delta = abs.(old .- rebuilt)

    println(rpad(name, 15),
        "max difference = ", maximum(delta),
        "; mean difference = ", mean(delta)
    )
end

# ---------------------------------------------------------
# Compare annual migration
# ---------------------------------------------------------

newmig = CSV.read(
    joinpath(newdir, "migration.csv"),
    DataFrame,
)

oldmig = vec(readdlm(
    joinpath(olddir, "migration"),
    Float64,
))

@assert length(oldmig) == nrow(newmig)

delta = abs.(oldmig .- newmig.mig)

println(rpad("migration", 15),
    "max difference = ", maximum(delta),
    "; mean difference = ", mean(delta)
)

println("-"^65)
println("Validation complete.")
