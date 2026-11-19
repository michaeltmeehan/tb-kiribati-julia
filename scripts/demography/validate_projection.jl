
using CSV
using DataFrames
using Statistics
using CairoMakie
using Test

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    input = joinpath(root, "data", "processed", "demography")
    outdir = joinpath(root, "results", "demography_validation")
    mkpath(outdir)

    load(name) = CSV.read(
        joinpath(input, "$name.csv"), DataFrame
    )

    population = load("population")
    fertility = load("fertility")
    mortality = load("mortality")
    migration = load("migration")
    vital = load("vital_totals")

    first_year, last_year = 2023, 2050
    ages = 0:100

    function annual_rows(df, year)
        d = filter(:year => ==(year), df)
        sort!(d, :age)
        return d
    end

    function age_vector(df, year, field)
        d = annual_rows(df, year)
        @assert d.age == collect(ages)
        return Float64.(d[!, field])
    end

    annual_value(df, year, field) = only(
        filter(:year => ==(year), df)[!, field]
    )

    wpp_population(y) =
        age_vector(population, y, :pop)

    # -------------------------------------------------------
    # Independent annual projection
    # -------------------------------------------------------

    function annual_step(
        N::AbstractVector,
        year::Int;
        female_fraction=0.485,
        use_wpp_births=false,
    )
        length(N) == length(ages) ||
            throw(DimensionMismatch("Invalid age vector"))

        μ = age_vector(mortality, year, :mxB)
        asfr = annual_rows(fertility, year).asfr
        net_migration = annual_value(
            migration, year, :migration
        )

        # Births from age-specific fertility.
        # Reproductive ages: 10–54 inclusive.
        births = use_wpp_births ?
                 annual_value(vital, year, :births) :
                 sum(
            N[11:55] .* female_fraction .* asfr
        )

        # Survival and whole-cohort ageing.
        survived = N .* exp.(-μ)

        updated = zeros(Float64, length(ages))

        updated[2:end] .= survived[1:(end-1)]

        # Open-ended age 100+.
        updated[end] += survived[end]

        # Approximate average newborn exposure of
        # half a year to infant mortality.
        updated[1] = births * exp(-μ[1] / 2)

        # Allocate net migration proportionally.
        before_migration = sum(updated)

        updated .+= net_migration .*
            updated ./ before_migration

        @test all(updated .>= 0)

        # Independent population accounting identity.
        @test isapprox(
            sum(updated),
            sum(survived) +
                births * exp(-μ[1] / 2) +
                net_migration;
            atol=1e-7
        )

        return updated
    end

    # -------------------------------------------------------
    # Run projection without reconciliation
    # -------------------------------------------------------

    function project(;
        female_fraction=0.485,
        use_wpp_births=false,
    )
        N = wpp_population(first_year)

        output = DataFrame(
            year=Int[],
            projected=Float64[],
            wpp=Float64[],
            relative_error=Float64[],
            age_distribution_distance=Float64[],
        )

        function record!(year, N)
            W = wpp_population(year)

            # Total variation distance between age shares.
            distance = 0.5 * sum(abs.(
                N ./ sum(N) .- W ./ sum(W)
            ))

            push!(output, (
                year,
                sum(N),
                sum(W),
                sum(N)/sum(W) - 1,
                distance,
            ))
        end

        record!(first_year, N)

        for year in (first_year+1):last_year
            N = annual_step(
                N, year;
                female_fraction,
                use_wpp_births,
            )
            record!(year, N)
        end

        return output, N
    end

    fixed485, N485 = project(female_fraction=0.485)
    fixed500, N500 = project(female_fraction=0.5)
    wppbirths, NWPP = project(use_wpp_births=true)

    # -------------------------------------------------------
    # Save numerical comparisons
    # -------------------------------------------------------

    for (name, df) in (
        ("fixed_0485", fixed485),
        ("fixed_0500", fixed500),
        ("wpp_births", wppbirths),
    )
        CSV.write(
            joinpath(outdir, "$name.csv"), df
        )
    end

    selected = [2023, 2024, 2025, 2030, 2040, 2050]

    println("\nINDEPENDENT DEMOGRAPHIC PROJECTION")

    for (name, df) in (
        ("Female fraction 0.485", fixed485),
        ("Female fraction 0.500", fixed500),
        ("WPP annual births", wppbirths),
    )
        println("\n", name)
        println(select(
            filter(:year => in(selected), df),
            :year, :projected, :wpp,
            :relative_error,
            :age_distribution_distance,
        ))
    end

    # -------------------------------------------------------
    # Plot total population errors
    # -------------------------------------------------------

    fig = Figure(size=(1000, 550))
    ax = Axis(
        fig[1, 1],
        xlabel="Year",
        ylabel="Population error (%)",
        title="Independent projection versus WPP2024",
    )

    for (df, label) in (
        (fixed485, "Female fraction 0.485"),
        (fixed500, "Female fraction 0.500"),
        (wppbirths, "WPP annual births"),
    )
        lines!(
            ax, df.year,
            100 .* df.relative_error;
            linewidth=2, label
        )
    end

    hlines!(ax, [0], color=:grey, linestyle=:dash)
    axislegend(ax)

    save(joinpath(outdir, "population_error.png"), fig)

    # -------------------------------------------------------
    # Plot final age distributions
    # -------------------------------------------------------

    fig = Figure(size=(1000, 550))
    ax = Axis(
        fig[1, 1],
        xlabel="Age",
        ylabel="Population",
        title="Population age distribution in 2050",
    )

    lines!(ax, collect(ages), wpp_population(2050);
        linewidth=2, label="WPP2024")

    lines!(ax, collect(ages), N485;
        linewidth=2, label="Independent projection")

    axislegend(ax)

    save(joinpath(outdir, "age_distribution_2050.png"), fig)

    println("\nOutputs saved to: ", outdir)
end
