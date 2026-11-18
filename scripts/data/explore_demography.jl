
using CSV
using DataFrames
using CairoMakie
using Statistics

let
    root = normpath(joinpath(@__DIR__, "..", ".."))

    demodir = joinpath(root, "data", "processed", "demography")
    survdir = joinpath(root, "data", "processed", "surveillance")
    outdir = joinpath(root, "results", "data_exploration")

    figdir = joinpath(outdir, "figures")
    tabdir = joinpath(outdir, "tables")

    mkpath(figdir)
    mkpath(tabdir)

    read_demo(name) =
        CSV.read(joinpath(demodir, "$name.csv"), DataFrame)

    read_surv(name) =
        CSV.read(joinpath(survdir, "$name.csv"), DataFrame)

    savefig(name, fig) =
        save(joinpath(figdir, "$name.png"), fig)

    savetab(name, df) =
        CSV.write(joinpath(tabdir, "$name.csv"), df)

    population = read_demo("population")
    fertility = read_demo("fertility")
    mortality = read_demo("mortality")
    migration = read_demo("migration")

    notifications = read_surv("tb_notifications_total")
    by_age = read_surv("tb_notifications_by_age")

    # -------------------------------------------------------
    # Population summaries
    # -------------------------------------------------------

    pop = combine(
        groupby(population, :year),
        :pop => sum => :total,
        :popF => sum => :female,
    )

    elderly = combine(
        groupby(filter(:age => >=(65), population), :year),
        :pop => sum => :age65plus,
    )

    children = combine(
        groupby(filter(:age => <(15), population), :year),
        :pop => sum => :age0to14,
    )

    pop = innerjoin(pop, elderly, on=:year)
    pop = innerjoin(pop, children, on=:year)
    sort!(pop, :year)

    pop.share65plus = pop.age65plus ./ pop.total
    pop.share0to14 = pop.age0to14 ./ pop.total

    savetab("demographic_summary", pop)

    # -------------------------------------------------------
    # 08. Population growth and ageing
    # -------------------------------------------------------

    fig = Figure(size=(1000, 750))

    ax1 = Axis(fig[1, 1],
        title="Kiribati population",
        ylabel="Population")

    lines!(ax1, pop.year, pop.total, linewidth=2)

    ax2 = Axis(fig[2, 1],
        xlabel="Year",
        ylabel="Percent of population",
        title="Population age structure")

    lines!(ax2, pop.year, 100 .* pop.share0to14,
        linewidth=2, label="Ages 0–14")

    lines!(ax2, pop.year, 100 .* pop.share65plus,
        linewidth=2, label="Ages 65+")

    axislegend(ax2)
    savefig("08_population_ageing", fig)

    # -------------------------------------------------------
    # 09. Fertility, mortality and migration
    # -------------------------------------------------------

    tfr = combine(
        groupby(fertility, :year),
        :tfr => first => :tfr
    )

    m0 = filter(:age => ==(0), mortality)
    m65 = filter(:age => ==(65), mortality)

    fig = Figure(size=(1000, 850))

    ax1 = Axis(fig[1, 1],
        ylabel="Children per woman",
        title="Total fertility rate")

    lines!(ax1, tfr.year, tfr.tfr, linewidth=2)

    ax2 = Axis(fig[2, 1],
        ylabel="Mortality hazard (per year)",
        title="Age-specific background mortality")

    lines!(ax2, m0.year, m0.mxB,
        linewidth=2, label="Age 0")

    lines!(ax2, m65.year, m65.mxB,
        linewidth=2, label="Age 65")

    axislegend(ax2)

    ax3 = Axis(fig[3, 1],
        xlabel="Year",
        ylabel="Net migrants per year",
        title="Net migration")

    lines!(ax3, migration.year,
        migration.migration, linewidth=2)

    hlines!(ax3, [0], color=(:grey, 0.5),
        linestyle=:dash)

    savefig("09_demographic_components", fig)

    # -------------------------------------------------------
    # Construct approximate midyear population
    # -------------------------------------------------------

    previous = select(
        population,
        :year, :age, :pop => :previous_population
    )
    previous.year .+= 1

    current = select(
        population,
        :year, :age, :pop => :current_population
    )

    midyear = innerjoin(
        previous, current, on=[:year, :age]
    )

    midyear.population = (
        midyear.previous_population .+
            midyear.current_population
    ) ./ 2

    # -------------------------------------------------------
    # Aggregate population to notification age groups
    # -------------------------------------------------------

    age_bounds = [
        ("0-4", 0, 5),
        ("5-14", 5, 15),
        ("0-14", 0, 15),
        ("15-24", 15, 25),
        ("25-34", 25, 35),
        ("35-44", 35, 45),
        ("45-54", 45, 55),
        ("55-64", 55, 65),
        ("65+", 65, 101),
    ]

    agepop = DataFrame(
        year=Int[],
        age_group=String[],
        population=Float64[],
    )

    for (label, lower, upper) in age_bounds
        selected = filter(
            :age => a -> lower <= a < upper,
            midyear
        )

        grouped = combine(
            groupby(selected, :year),
            :population => sum => :population
        )

        for r in eachrow(grouped)
            push!(agepop, (
                r.year, label, r.population
            ))
        end
    end

    # -------------------------------------------------------
    # Age-specific notification rates
    # -------------------------------------------------------

    rates = innerjoin(
        by_age, agepop,
        on=[:year, :age_group]
    )

    @assert nrow(rates) == nrow(by_age)

    rates.rate_per_100k =
        1e5 .* rates.cases ./ rates.population

    sort!(rates, [:year, :age_group])

    savetab("notification_rates_by_age", rates)

    # -------------------------------------------------------
    # 10. Age-specific notification rates
    # -------------------------------------------------------

    age_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+"
    ]

    fig = Figure(size=(1000, 800))

    for (panel, labels) in enumerate((
        age_labels[1:4],
        age_labels[5:8],
    ))
        ax = Axis(fig[panel, 1],
            title=panel == 1 ? "Ages 0–34" : "Ages 35+",
            xlabel=panel == 2 ? "Year" : "",
            ylabel="Notifications per 100,000")

        for (j, label) in enumerate(labels)
            d = filter(:age_group => ==(label), rates)
            sort!(d, :year)

            years = collect(minimum(d.year):maximum(d.year))

            values = [
                begin
                    idx = findfirst(==(y), d.year)
                    isnothing(idx) ? NaN :
                    d.rate_per_100k[idx]
                end
                for y in years
            ]

            lines!(ax, years, values,
                color=Cycled(j),
                linewidth=2, label=label)

            scatter!(ax, d.year, d.rate_per_100k,
                color=Cycled(j), markersize=6)
        end

        axislegend(ax, position=:rt)
    end

    savefig("10_notification_rates_by_age", fig)

    # -------------------------------------------------------
    # National notification rates
    # -------------------------------------------------------

    national_pop = combine(
        groupby(midyear, :year),
        :population => sum => :population
    )

    national = innerjoin(
        notifications,
        national_pop,
        on=:year
    )

    national.rate_per_100k =
        1e5 .* national.cases ./ national.population

    sort!(national, :year)
    savetab("notification_rates_national", national)

    # -------------------------------------------------------
    # 11. National counts versus rates
    # -------------------------------------------------------

    fig = Figure(size=(1000, 700))

    ax1 = Axis(fig[1, 1],
        title="National TB notifications",
        ylabel="Annual notifications")

    lines!(ax1, national.year, national.cases,
        linewidth=2)

    ax2 = Axis(fig[2, 1],
        xlabel="Year",
        ylabel="Notifications per 100,000")

    lines!(ax2, national.year,
        national.rate_per_100k, linewidth=2)

    savefig("11_notification_counts_rates", fig)

    println("\nDEMOGRAPHIC SUMMARY")
    println(filter(
        :year => in([1950, 2000, 2023, 2025, 2050, 2100]),
        pop
    ))

    println("\nAGE-SPECIFIC NOTIFICATION RATES")
    println(first(rates, 12))

    println("\nSaved figures to: ", figdir)
    println("Saved tables to: ", tabdir)
end
