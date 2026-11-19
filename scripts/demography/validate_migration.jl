
using CSV
using DataFrames
using Statistics
using Test
using CairoMakie

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

    first_year, last_year = 2023, 2050
    ages = 0:100
    female_fraction = 0.5

    function age_values(df, year, field)
        rows = sort(
            filter(:year => ==(year), df), :age
        )
        @assert rows.age == collect(ages)
        Float64.(rows[!, field])
    end

    function annual_value(df, year, field)
        rows = filter(:year => ==(year), df)
        @assert nrow(rows) == 1
        Float64(rows[1, field])
    end

    reference(year) = age_values(pop, year, :pop)


    function migration(year, scheme)
        national = annual_value(
            mig_total, year, :migration
        )

        if scheme == :age
            M = 1000 .* age_values(mig_age, year, :mig)

            # Reconcile the small difference between
            # age-specific and national migration totals.
            discrepancy = national - sum(M)

            W = reference(year - 1)
            M .+= discrepancy .* W ./ sum(W)

            @assert isapprox(
                sum(M), national; atol=1e-8
            )

            return M

        elseif scheme == :proportional
            return national
        else
            throw(ArgumentError(
                "Unknown migration scheme: $scheme"
            ))
        end
    end


    function step(N, year; scheme=:age)
        μ = age_values(mort, year, :mxB)
        fert_year = sort(
            filter(:year => ==(year), fert), :age
        )

        @assert fert_year.age == collect(10:54)

        asfr = zeros(Float64, length(ages))
        asfr[11:55] .= fert_year.asfr

        # Fertility uses the model population,
        # not WPP's female population.
        births = female_fraction * sum(N .* asfr)

        survived = N .* exp.(-μ)

        next = zeros(Float64, length(N))
        next[2:end] .= survived[1:(end-1)]
        next[end] += survived[end]

        next[1] = births * exp(-μ[1] / 2)

        deaths = sum(N) + births - sum(next)

        M = migration(year, scheme)

        if scheme == :age
            next .+= M
            migration_total = sum(M)
        else
            migration_total = M
            next .+= M .* next ./ sum(next)
        end

        @assert all(next .>= 0)

        @test isapprox(
            sum(next),
            sum(N) + births - deaths + migration_total;
            atol=1e-7,
        )

        return next, births
    end

    function diagnostics(N, year, scheme, mode, births)
        W = reference(year)

        return (
            scheme=String(scheme),
            mode=String(mode),
            year=year,
            total_error_pct=
            100 * (sum(N) / sum(W) - 1),
            age_TVD_pct=
            50 * sum(abs.(
                N ./ sum(N) .- W ./ sum(W)
            )),
            age_MAE=mean(abs.(N .- W)),
            age_max_error=maximum(abs.(N .- W)),
            births=births,
            wpp_births=annual_value(
                vital, year, :births
            ),
        )
    end


    const_age_breaks = [0, 5, 15, 25, 35, 45, 55, 65, 101]

    const_age_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+",
    ]

    function age_group_diagnostics(N, year, scheme, mode)
        W = reference(year)

        rows = NamedTuple[]

        for g in eachindex(const_age_labels)
            lo = const_age_breaks[g]
            hi = const_age_breaks[g+1]

            # Age a corresponds to vector index a+1.
            indices = (lo+1):hi

            projected = sum(@view N[indices])
            reference_pop = sum(@view W[indices])

            push!(rows, (
                scheme=String(scheme),
                mode=String(mode),
                year=year,
                age_group=const_age_labels[g],
                projected=projected,
                wpp=reference_pop,
                error_pct=100 * (
                    projected / reference_pop - 1
                ),
            ))
        end

        return rows
    end


    # Validate migration totals against WPP.
    migration_check = DataFrame(
        year=Int[],
        age_sum=Float64[],
        national=Float64[],
        difference=Float64[],
    )

    for year in (first_year+1):last_year
        age_sum = sum(
            1000 .* age_values(mig_age, year, :mig)
        )
        national = annual_value(
            mig_total, year, :migration
        )

        push!(migration_check, (
            year,
            age_sum,
            national,
            age_sum - national,
        ))
    end

    println("\nMIGRATION CONSISTENCY")
    show(stdout, MIME"text/plain"(), migration_check; allrows=true)
    println()

    println(
        "\nMaximum absolute discrepancy: ",
        maximum(abs.(migration_check.difference)),
        " people"
    )

    CSV.write(
        joinpath(outdir, "migration_consistency.csv"),
        migration_check,
    )


    results = DataFrame()
    age_results = DataFrame()

    for scheme in (:proportional, :age)

        # One-year predictions: reset to WPP every year.
        for year in (first_year+1):last_year
            N, births = step(
                reference(year-1), year; scheme
            )

            push!(
                results,
                diagnostics(
                    N, year, scheme, :one_year, births
                ),
            )

            append!(
                age_results,
                age_group_diagnostics(
                    N, year, scheme, :one_year
                );
                cols=:union,
            )
        end

        # Cumulative projection: initialise only once.
        N = reference(first_year)

        for year in (first_year+1):last_year
            N, births = step(N, year; scheme)

            push!(
                results,
                diagnostics(
                    N, year, scheme, :cumulative, births
                ),
            )

            append!(
                age_results,
                age_group_diagnostics(
                    N, year, scheme, :cumulative
                );
                cols=:union,
            )
        end
    end

    CSV.write(
        joinpath(outdir, "migration_diagnostics.csv"),
        results,
    )

    selected = [2024, 2025, 2030, 2040, 2050]

    println("\nONE-YEAR PREDICTIONS")
    println(
        filter(
            row -> row.year in selected &&
                row.mode == "one_year",
            results,
        )[:, [
            :year, :scheme,
            :total_error_pct, :age_TVD_pct,
            :births, :wpp_births,
        ]]
    )

    println("\nCUMULATIVE PROJECTIONS")
    println(
        filter(
            row -> row.year in selected &&
                row.mode == "cumulative",
            results,
        )[:, [
            :year, :scheme,
            :total_error_pct, :age_TVD_pct,
        ]]
    )

    # Compare one-year and cumulative errors.
    fig = Figure(size=(1000, 650))

    ax1 = Axis(
        fig[1, 1],
        title="Total population error",
        ylabel="Error (%)",
    )

    ax2 = Axis(
        fig[2, 1],
        xlabel="Year",
        ylabel="Age-distribution TVD (%)",
    )

    for scheme in ("proportional", "age")
        for mode in ("one_year", "cumulative")
            d = filter(
                row -> row.scheme == scheme &&
                    row.mode == mode,
                results,
            )

            label = "$scheme / $mode"

            lines!(
                ax1, d.year, d.total_error_pct;
                label,
            )

            lines!(
                ax2, d.year, d.age_TVD_pct;
                label,
            )
        end
    end

    axislegend(ax1; position=:rb)

    save(
        joinpath(outdir, "migration_diagnostics.png"),
        fig,
    )


    CSV.write(
        joinpath(outdir, "age_group_diagnostics.csv"),
        age_results,
    )

    selected_years = [2030, 2040, 2050]

    selected_ages = filter(
        row -> row.year in selected_years,
        age_results,
    )

    summary = combine(
        groupby(
            selected_ages,
            [:scheme, :mode, :year],
        ),
        :error_pct => (
            x -> maximum(abs.(x))
        ) => :max_absolute_error_pct,
    )

    sort!(summary, [:mode, :year, :scheme])

    println("\nMAXIMUM ABSOLUTE AGE-GROUP ERROR (%)")
    show(stdout, MIME"text/plain"(), summary; allrows=true)
    println()

    println("\nAGE-GROUP ERRORS IN 2050 (%)")

    final_ages = filter(
        row -> row.year == 2050 &&
            row.mode == "cumulative",
        age_results,
    )

    wide = unstack(
        final_ages,
        :age_group,
        :scheme,
        :error_pct,
    )

    show(stdout, MIME"text/plain"(), wide; allrows=true)
    println()

    println("\nResults saved to ", outdir)
end
