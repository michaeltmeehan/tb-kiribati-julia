
module PremReciprocity

using CSV
using DataFrames
using DelimitedFiles

export PremContacts, source_matrix, contact_matrix,
    reciprocity_error

# Original Prem age bands:
# 0-4, 5-9, ..., 70-74, 75+
band(age::Integer) = min(div(age, 5) + 1, 16)

struct PremContacts
    source::Matrix{Float64}
    years::Vector{Int}
    populations::Matrix{Float64}  # Years × 16 groups
end

function PremContacts(source_path::AbstractString,
    population_path::AbstractString)

    C = Matrix{Float64}(readdlm(source_path, ','))
    size(C) == (16, 16) ||
        throw(DimensionMismatch("Expected 16×16 Prem matrix"))

    all(x -> isfinite(x) && x >= 0, C) ||
        throw(ArgumentError("Invalid Prem contacts"))

    df = CSV.read(population_path, DataFrame)

    allunique(df[:, [:year, :age]]) ||
        error("Duplicate population year-age observations")

    years = sort(unique(df.year))
    P = zeros(length(years), 16)

    for (i, y) in enumerate(years)
        d = filter(:year => ==(y), df)

        sort(d.age) == collect(0:100) ||
            error("Incomplete population age grid: $y")

        for r in eachrow(d)
            isfinite(r.pop) && r.pop >= 0 ||
                error("Invalid population: $y")

            P[i, band(r.age)] += r.pop
        end
    end

    all(>(0), P) ||
        error("Every contact age group needs positive population")

    return PremContacts(C, years, P)
end

# Annual population with linear interpolation for fractional years.
function population_at(prem::PremContacts, year::Real)
    isfinite(year) || throw(ArgumentError("Invalid year"))

    first(prem.years) <= year <= last(prem.years) ||
        throw(ArgumentError("Year outside WPP coverage"))

    j = searchsortedlast(prem.years, year)

    if j == length(prem.years) ||
        prem.years[j] == year
        return copy(vec(prem.populations[j, :]))
    end

    y0, y1 = prem.years[j:(j+1)]
    α = (year - y0) / (y1 - y0)

    return (1 - α) .* vec(prem.populations[j, :]) .+
        α .* vec(prem.populations[j+1, :])
end

function source_matrix(prem::PremContacts, year::Real)
    N = population_at(prem, year)
    C = prem.source
    adjusted = similar(C)

    for i in 1:16, j in 1:16
        adjusted[i, j] =
            (N[i] * C[i, j] + N[j] * C[j, i]) /
                (2 * N[i])
    end

    return adjusted
end

function reciprocity_error(C, N)
    size(C) == (length(N), length(N)) ||
        throw(DimensionMismatch("Incompatible dimensions"))

    flows = N .* C
    return maximum(abs.(flows .- transpose(flows)))
end

# Expand 16 source bands to single-year ages 0:95.
# Last model age (95) represents 95+.
function expand_contacts(C::AbstractMatrix; nages::Int=96)
    size(C) == (16, 16) ||
        throw(DimensionMismatch("Expected 16×16 matrix"))

    nages >= 76 ||
        throw(ArgumentError("At least 76 model ages required"))

    groups = [band(a) for a in 0:(nages-1)]
    sizes = [count(==(g), groups) for g in 1:16]

    return [
        C[groups[i], groups[j]] / sizes[groups[j]]
        for i in 1:nages, j in 1:nages
    ]
end

function contact_matrix(
    prem::PremContacts,
    year::Real;
    reciprocal::Bool=true,
    nages::Int=96,
)
    C = reciprocal ?
        source_matrix(prem, year) : prem.source

    return expand_contacts(C; nages=nages)
end

end
