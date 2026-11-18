
module WPP2024Processing

using DataFrames
using Statistics

export process_demography

const YEARS = 1950:2100

function check_grid(df, ages)
    allunique(df[:, [:year, :age]]) ||
        error("Duplicate demographic year-age records")

    for year in YEARS
        d = df[df.year .== year, :]
        sort(d.age) == collect(ages) ||
            error("Incomplete demographic age grid: $year")
    end

    sort(unique(df.year)) == collect(YEARS) ||
        error("Unexpected demographic years")

    return nothing
end

function process_demography(
    population_raw,
    fertility_raw,
    mortality_raw,
    migration_raw,
    vital_raw,
)
    # Do not modify caller-owned data.
    population = copy(population_raw)
    fertility = copy(fertility_raw)
    mortality = copy(mortality_raw)
    migration = copy(migration_raw)
    vital = copy(vital_raw)

    # Population: thousands -> persons.
    filter!(:year => in(YEARS), population)
    sort!(population, [:year, :age])
    check_grid(population, 0:100)

    for c in (:popM, :popF, :pop)
        population[!, c] .*= 1000
    end

    all(isfinite, population.pop) &&
        all(population.pop .>= 0) ||
        error("Invalid population")

    maximum(abs.(
        population.pop .-
            population.popM .-
            population.popF
    )) <= 1.01 ||
        error("Inconsistent population sex totals")

    # Fertility: percentASFR and TFR -> ASFR.
    sort!(fertility, [:year, :age])
    check_grid(fertility, 10:54)

    fertility.asfr =
        fertility.tfr .* fertility.pasfr ./ 100

    all(isfinite, fertility.asfr) &&
        all(fertility.asfr .>= 0) ||
        error("Invalid fertility rates")

    fertility_checks = combine(
        groupby(fertility, :year),
        :pasfr => sum => :percent_sum,
        :asfr => sum => :tfr_reconstructed,
        :tfr => first => :tfr_reported,
    )

    maximum(abs.(
        fertility_checks.percent_sum .- 100
    )) < 0.1 ||
        error("Fertility percentages do not sum to 100")

    # Mortality: annual central death rates.
    sort!(mortality, [:year, :age])
    check_grid(mortality, 0:100)

    all(isfinite, mortality.mxB) &&
        all(mortality.mxB .>= 0) ||
        error("Invalid mortality rates")

    # Migration: thousands -> persons.
    sort!(migration, :year)

    migration.year == collect(YEARS) ||
        error("Incomplete migration years")

    migration.migration = migration.mig .* 1000

    # Auxiliary demographic totals.
    filter!(:year => in(YEARS), vital)
    sort!(vital, :year)

    for c in (:births, :deaths, :PopChange)
        vital[!, c] .*= 1000
    end

    # Annual population totals.
    population_totals = combine(
        groupby(population, :year),
        :pop => sum => :population,
    )

    return (
        population=population,
        fertility=fertility,
        mortality=mortality,
        migration=migration,
        vital_totals=vital,
        fertility_checks=fertility_checks,
        population_totals=population_totals,
    )
end

end # module
