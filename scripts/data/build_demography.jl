
using CSV
using DataFrames
using Statistics

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    rawdir = joinpath(root, "data", "raw", "wpp2024")
    outdir = joinpath(root, "data", "processed", "demography")
    mkpath(outdir)

    readraw(name) =
        CSV.read(joinpath(rawdir, "$name.csv"), DataFrame)

    function check_grid(df, ages; years=1950:2100)
        allunique(df[:, [:year, :age]]) ||
            error("Duplicate year-age records")

        for year in years
            d = df[df.year .== year, :]
            sort(d.age) == collect(ages) ||
                error("Incomplete age grid in $year")
        end

        return nothing
    end

    # Population
    population = readraw("population")
    filter!(:year => in(1950:2100), population)
    sort!(population, [:year, :age])

    check_grid(population, 0:100)

    # Convert thousands to persons.
    for c in (:popM, :popF, :pop)
        population[!, c] .*= 1000
    end

    all(population.pop .>= 0) ||
        error("Negative population")

    maximum(abs.(
        population.pop .-
            population.popM .-
            population.popF
    )) <= 1.01 ||
        error("Population sex totals inconsistent")

    # Fertility
    fertility = readraw("fertility")
    sort!(fertility, [:year, :age])
    check_grid(fertility, 10:54)

    # WPP percentASFR is a percentage distribution.
    # The resulting ASFR is births per female person-year.
    fertility.asfr =
        fertility.tfr .* fertility.pasfr ./ 100

    all(fertility.asfr .>= 0) ||
        error("Negative fertility rate")

    fert_checks = combine(
        groupby(fertility, :year),
        :pasfr => sum => :percent_sum,
        :asfr => sum => :tfr_reconstructed,
        :tfr => first => :tfr_reported
    )

    maximum(abs.(fert_checks.percent_sum .- 100)) < 0.1 ||
        error("Fertility percentages do not sum to 100")

    # Mortality
    mortality = readraw("mortality")
    sort!(mortality, [:year, :age])
    check_grid(mortality, 0:100)

    all(mortality.mxB .>= 0) ||
        error("Negative mortality rate")

    # Migration
    migration = readraw("migration")
    sort!(migration, :year)

    migration.year == collect(1950:2100) ||
        error("Incomplete migration years")

    # Thousands of migrants to persons per year.
    migration.migration = migration.mig .* 1000

    # Useful auxiliary demographic series
    vital = readraw("vital_totals")
    filter!(:year => in(1950:2100), vital)
    sort!(vital, :year)

    # Preserve WPP fields but convert counts to persons.
    for c in (:births, :deaths, :PopChange)
        vital[!, c] .*= 1000
    end

    # Write processed datasets
    datasets = (
        population=population,
        fertility=fertility,
        mortality=mortality,
        migration=migration,
        vital_totals=vital,
    )

    for (name, df) in pairs(datasets)
        CSV.write(joinpath(outdir, "$name.csv"), df)
        println(rpad(string(name), 16), nrow(df), " rows")
    end

    CSV.write(
        joinpath(outdir, "fertility_checks.csv"),
        fert_checks
    )

    # Population trajectory
    totals = combine(
        groupby(population, :year),
        :pop => sum => :population
    )

    CSV.write(
        joinpath(outdir, "population_totals.csv"),
        totals
    )

    println("\nSelected population totals:")
    println(filter(:year => in([1950, 2000, 2023, 2025, 2050, 2100]),
        totals))

    println("\nMaximum TFR reconstruction error: ",
        maximum(abs.(
            fert_checks.tfr_reconstructed .-
                fert_checks.tfr_reported
        ))
    )
end
