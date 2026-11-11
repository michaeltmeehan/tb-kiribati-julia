using DelimitedFiles: readdlm

abstract type AbstractDemography end

"""
Demographic inputs for Kiribati.

Rows of population, fertility and mortality matrices correspond
to single-year ages. Columns correspond to calendar years.

Population and migration data are stored in thousands in the
source files and converted to individuals when retrieved.
"""
struct DemographicData
    population::Matrix{Float64}
    fertility::Matrix{Float64}
    mortality::Matrix{Float64}
    migration::Vector{Float64}

    first_year::Int
    last_year::Int
    first_fertility_age::Int
    female_fraction::Float64
end


function load_demographic_data(
    directory::AbstractString;
    first_year::Int = 1950,
    first_fertility_age::Int = 10,
    female_fraction::Real = 0.485,
)
    population = readdlm(
        joinpath(directory, "population"),
        Float64,
    )

    fertility = readdlm(
        joinpath(directory, "fertility"),
        Float64,
    )

    mortality = readdlm(
        joinpath(directory, "mortality"),
        Float64,
    )

    migration = vec(readdlm(
        joinpath(directory, "migration"),
        Float64,
    ))

    nyears = size(population, 2)

    size(fertility, 2) == nyears ||
        throw(DimensionMismatch("Fertility year count differs"))

    size(mortality, 2) == nyears ||
        throw(DimensionMismatch("Mortality year count differs"))

    length(migration) == nyears ||
        throw(DimensionMismatch("Migration year count differs"))

    size(population, 1) >= 96 ||
        throw(DimensionMismatch(
            "Population data must contain at least 96 ages"
        ))

    size(mortality, 1) >= 96 ||
        throw(DimensionMismatch(
            "Mortality data must contain at least 96 ages"
        ))

    0 <= female_fraction <= 1 ||
        throw(ArgumentError(
            "female_fraction must lie in [0, 1]"
        ))

    all(isfinite, population) ||
        throw(ArgumentError("Nonfinite population data"))

    all(isfinite, fertility) ||
        throw(ArgumentError("Nonfinite fertility data"))

    all(isfinite, mortality) ||
        throw(ArgumentError("Nonfinite mortality data"))

    all(isfinite, migration) ||
        throw(ArgumentError("Nonfinite migration data"))

    all(>=(0), population) ||
        throw(ArgumentError("Negative population data"))

    all(>=(0), fertility) ||
        throw(ArgumentError("Negative fertility data"))

    all(>=(0), mortality) ||
        throw(ArgumentError("Negative mortality data"))

    return DemographicData(
        population,
        fertility,
        mortality,
        migration,
        first_year,
        first_year + nyears - 1,
        first_fertility_age,
        Float64(female_fraction),
    )
end


# ------------------------------------------------------------
# Year indexing
# ------------------------------------------------------------

@inline function demographic_year_index(
    data::DemographicData,
    year::Real,
)
    isfinite(year) ||
        throw(ArgumentError("Year must be finite"))

    y = clamp(
        floor(Int, year),
        data.first_year,
        data.last_year,
    )

    return y - data.first_year + 1
end


# ------------------------------------------------------------
# Population
# ------------------------------------------------------------

"""
    get_population(data, year; nages=96)

Return population by model age class.

The final age class contains everyone aged nages-1 or older.
"""
function get_population(
    data::DemographicData,
    year::Real;
    nages::Int = 96,
)
    2 <= nages <= size(data.population, 1) ||
        throw(ArgumentError("Invalid number of age classes"))

    col = demographic_year_index(data, year)

    result = Vector{Float64}(undef, nages)

    @views result[1:end-1] .=
        1000 .* data.population[1:nages-1, col]

    result[end] =
        1000 * sum(@view data.population[nages:end, col])

    return result
end


# ------------------------------------------------------------
# Fertility
# ------------------------------------------------------------

@inline function fertility_rate(
    data::DemographicData,
    age::Int,
    year::Real,
)
    row = age - data.first_fertility_age + 1

    1 <= row <= size(data.fertility, 1) ||
        return 0.0

    col = demographic_year_index(data, year)

    return data.female_fraction * data.fertility[row, col]
end


# ------------------------------------------------------------
# Background mortality
# ------------------------------------------------------------

@inline function mortality_rate(
    data::DemographicData,
    age::Int,
    year::Real,
)
    age >= 0 ||
        throw(ArgumentError("Age must be nonnegative"))

    row = min(age + 1, size(data.mortality, 1))
    col = demographic_year_index(data, year)

    return data.mortality[row, col]
end


# ------------------------------------------------------------
# Net migration
# ------------------------------------------------------------

@inline function migration_flow(
    data::DemographicData,
    year::Real,
)
    col = demographic_year_index(data, year)

    return 1000 * data.migration[col]
end