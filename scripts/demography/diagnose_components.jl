
using CSV
using DataFrames
using Statistics
using CairoMakie
using Test

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    processed = joinpath(root, "data", "processed", "demography")
    raw = joinpath(root, "data", "raw", "wpp2024")
    outdir = joinpath(root, "results", "demography_validation")
    mkpath(outdir)

    read_processed(name) = CSV.read(
        joinpath(processed, "$name.csv"), DataFrame
    )

    pop = read_processed("population")
    fert = read_processed("fertility")
    mort = read_processed("mortality")
    mig_total = read_processed("migration")
    vital = read_processed("vital_totals")
    mig_age = CSV.read(
        joinpath(raw, "migration_age.csv"), DataFrame
    )

    ages = 0:100
    first_year, last_year = 2023, 2050
    female_fraction = 0.5

    function age_values(df, year, field; age_range=ages)
        rows = sort(filter(:year => ==(year), df), :age)
        @assert collect(rows.age) == collect(age_range)
        return Float64.(rows[!, field])
    end

    function annual_value(df, year, field)
        rows = filter(:year => ==(year), df)
        @assert nrow(rows) == 1
        return Float64(rows[1, field])
    end

    reference(year) = age_values(pop, year, :pop)

    function fertility_rates(year)
        rates = zeros(length(ages))
        rates[11:55] .= age_values(
            fert, year, :asfr; age_range=10:54
        )
        return rates
    end

    function migration_vector(year, scheme, N)
        national = annual_value(
            mig_total, year, :migration
        )

        if scheme == :age
            M = 1000 .* age_values(
                mig_age, year, :mig
            )

            # Adjust age-specific totals without
            # changing the national net migration.
            M .+= (national - sum(M)) .* N ./ sum(N)
            return M

        elseif scheme == :proportional
            return national
        end

        error("Unknown migration scheme")
    end

    function step(N, year;
        scheme=:age,
        fertility_method=:start)

        μ = age_values(mort, year, :mxB)
        f = fertility_rates(year)

        # Background mortality.
        survived = N .* exp.(-μ)

        # Whole-cohort ageing.
        next = zeros(length(N))
        next[2:end] .= survived[1:(end-1)]
        next[end] += survived[end]

        # Migration before adding newborns.
        M = migration_vector(year, scheme, N)

        if scheme == :age
            next .+= M
            migration_total = sum(M)
        else
            migration_total = M
            next .+= M .* next ./ sum(next)
        end

        # Fertility exposure: either start-year or
        # mean of start-year and pre-birth end-year.
        exposure = if fertility_method == :start
            N
        elseif fertility_method == :midyear
            (N .+ next) ./ 2
        else
            error("Unknown fertility method")
        end

        births = female_fraction * sum(exposure .* f)

        # Approximate infant survival over half a year.
        surviving_births = births * exp(-μ[1] / 2)
        next[1] += surviving_births

        # Background deaths plus newborn deaths.
        deaths = sum(N) - sum(survived) +
            births - surviving_births

        @assert all(isfinite, next)
        @assert all(next .>= 0)

        @test isapprox(
            sum(next),
            sum(N) + births - deaths + migration_total;
            atol=1e-7
        )

        return (
            population=next,
            births=births,
            deaths=deaths,
            migration=migration_total,
        )
    end

    # --------------------------------------------------
    # Annual component and projection diagnostics
    # --------------------------------------------------

    records = NamedTuple[]
    ages_records = NamedTuple[]

    age_breaks = [0, 5, 15, 25, 35, 45, 55, 65, 101]
    age_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+"
    ]

    function record!(year, result, scheme, method, mode)
        N = result.population
        W = reference(year)

        B = annual_value(vital, year, :births)
        D = annual_value(vital, year, :deaths)
        M = annual_value(mig_total, year, :migration)

        # WPP's own rounded demographic-accounting residual.
        Wprev = reference(year - 1)
        balance_residual =
            B - D + M - (sum(W) - sum(Wprev))

        population_error = sum(N) - sum(W)

        push!(records, (
            year=year,
            scheme=String(scheme),
            fertility=String(method),
            mode=String(mode),
            births_error=result.births - B,
            deaths_error=result.deaths - D,
            migration_error=result.migration - M,
            wpp_balance_residual=balance_residual,
            population_error_pct=
            100 * population_error / sum(W),
            age_TVD_pct=
            50 * sum(abs.(
                N ./ sum(N) .- W ./ sum(W)
            )),
        ))

        for g in eachindex(age_labels)
            lo = age_breaks[g]
            hi = age_breaks[g+1]
            inds = (lo+1):hi

            actual = sum(@view W[inds])
            predicted = sum(@view N[inds])

            push!(ages_records, (
                year=year,
                scheme=String(scheme),
                fertility=String(method),
                mode=String(mode),
                age_group=age_labels[g],
                error_pct=100 * (predicted / actual - 1),
            ))
        end

        # For one-year predictions, the error must
        # equal the sum of component discrepancies
        # plus the WPP accounting residual.
        if mode == :one_year
            explained =
                (result.births - B) -
                        (result.deaths - D) +
                    (result.migration - M) +
                    balance_residual

            @test isapprox(
                population_error, explained;
                atol=1e-7
            )
        end
    end

    for scheme in (:proportional, :age)
        for method in (:start, :midyear)

            # One-year forecasts.
            for year in (first_year+1):last_year
                result = step(
                    reference(year-1), year;
                    scheme,
                    fertility_method=method,
                )

                record!(
                    year, result, scheme, method, :one_year
                )
            end

            # Cumulative projections.
            N = reference(first_year)

            for year in (first_year+1):last_year
                result = step(
                    N, year;
                    scheme,
                    fertility_method=method,
                )

                N = result.population

                record!(
                    year, result, scheme, method, :cumulative
                )
            end
        end
    end

    results = DataFrame(records)
    age_results = DataFrame(ages_records)

    CSV.write(
        joinpath(outdir, "component_diagnostics.csv"),
        results,
    )
    CSV.write(
        joinpath(outdir, "component_age_groups.csv"),
        age_results,
    )

    # --------------------------------------------------
    # Summaries
    # --------------------------------------------------

    println("\nONE-YEAR COMPONENT ERRORS (PERSONS)")

    one_year = filter(
        row -> row.mode == "one_year" &&
            row.scheme == "age" &&
            row.year in [2024, 2030, 2040, 2050],
        results,
    )

    show(
        stdout, MIME"text/plain"(),
        select(
            one_year,
            :year, :fertility,
            :births_error, :deaths_error,
            :migration_error,
            :wpp_balance_residual,
            :population_error_pct,
        );
        allrows=true,
    )
    println()

    println("\nCUMULATIVE PROJECTION ERRORS")

    cumulative = filter(
        row -> row.mode == "cumulative" &&
            row.year in [2030, 2040, 2050],
        results,
    )

    show(
        stdout, MIME"text/plain"(),
        select(
            cumulative,
            :year, :scheme, :fertility,
            :population_error_pct, :age_TVD_pct,
        );
        allrows=true,
    )
    println()

    println("\nMAXIMUM ABSOLUTE AGE-GROUP ERROR (%)")

    summary = combine(
        groupby(
            filter(
                row -> row.mode == "cumulative" &&
                    row.year == 2050,
                age_results,
            ),
            [:scheme, :fertility],
        ),
        :error_pct =>
            (x -> maximum(abs.(x))) =>
                :max_error_pct,
    )

    show(
        stdout, MIME"text/plain"(), summary;
        allrows=true,
    )
    println()

    # --------------------------------------------------
    # Figures
    # --------------------------------------------------

    fig = Figure(size=(1000, 650))

    ax1 = Axis(
        fig[1, 1],
        ylabel="Population error (%)",
        title="Cumulative demographic projection",
    )

    ax2 = Axis(
        fig[2, 1],
        xlabel="Year",
        ylabel="Age-distribution TVD (%)",
    )

    for scheme in ("proportional", "age")
        for method in ("start", "midyear")
            d = filter(
                row -> row.mode == "cumulative" &&
                    row.scheme == scheme &&
                    row.fertility == method,
                results,
            )

            label = "$scheme / $method"

            lines!(
                ax1, d.year, d.population_error_pct;
                label,
            )
            lines!(
                ax2, d.year, d.age_TVD_pct;
                label,
            )
        end
    end

    axislegend(ax1)

    save(
        joinpath(outdir, "component_projection.png"),
        fig,
    )

    println("\nSaved results to: ", outdir)
end
