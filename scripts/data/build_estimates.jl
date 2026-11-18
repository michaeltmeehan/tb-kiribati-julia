
using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(ROOT, "src", "data", "who2026.jl"))

rawdir = joinpath(
    ROOT, "data", "raw", "who", "2026", "estimates"
)

outdir = joinpath(
    ROOT, "data", "processed", "surveillance", "estimates"
)

result = WHO2026.build_estimates(rawdir, outdir)

for (name, table) in pairs(result)
    println(rpad(string(name), 26), nrow(table), " rows")
end

# Extract the national 2025 incidence estimate.
national = filter(
    r -> r.age_group == "all" &&
        r.sex == "a" &&
        r.risk_factor == "all",
    result.incidence_age_sex,
)

println("\nNational incidence estimate:")
println(national)

# Extract model-compatible age groups, without aggregation.
ages = [
    "0-4", "5-14", "15-24", "25-34",
    "35-44", "45-54", "55-64", "65plus",
]

age_incidence = filter(
    r -> r.age_group in ages &&
        r.sex == "a" &&
        r.risk_factor == "all",
    result.incidence_age_sex,
)

println("\nAge-specific incidence estimates:")
println(select(age_incidence, :age_group, :best, :lo, :hi))


country = WHO2026.build_country_burden(
    rawdir, outdir
)

println("\nNational burden: ", nrow(country), " years")

println(select(
    filter(:year => >=(2020), country),
    :year,
    :e_inc_num,
    :e_inc_num_lo,
    :e_inc_num_hi,
    :e_mort_num,
    :e_mort_num_lo,
    :e_mort_num_hi,
    :c_cdr,
))

