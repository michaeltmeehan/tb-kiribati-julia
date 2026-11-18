
using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(ROOT, "src", "data", "who2026.jl"))

rawdir = joinpath(ROOT, "data", "raw", "who", "2026", "reported")
outdir = joinpath(ROOT, "data", "processed", "surveillance")

result = WHO2026.build_surveillance(rawdir, outdir)

for (name, table) in pairs(result)
    println(rpad(string(name), 38), nrow(table), " rows")
end

println("\nNotification discrepancies:")
println(filter(
    :difference => x -> !ismissing(x) && x != 0,
    result.notification_checks,
))

println("\nTreatment-outcome discrepancies:")
println(filter(
    :difference => x -> !ismissing(x) && x != 0,
    result.treatment_checks,
))
