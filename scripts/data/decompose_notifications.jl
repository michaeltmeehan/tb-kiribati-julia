
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

    population = CSV.read(
        joinpath(demodir, "population.csv"), DataFrame
    )
    notifications = CSV.read(
        joinpath(survdir, "tb_notifications_by_age.csv"),
        DataFrame
    )

    # -------------------------------------------------------
    # Consistent seven-group age structure
    # -------------------------------------------------------

    age_groups = [
        ("0-14", 0, 15),
        ("15-24", 15, 25),
        ("25-34", 25, 35),
        ("35-44", 35, 45),
        ("45-54", 45, 55),
        ("55-64", 55, 65),
        ("65+", 65, 101),
    ]

    years = 2013:2025

    # Approximate midyear populations
    previous = select(
        population,
        :year, :age, :pop => :previous
    )
    previous.year .+= 1

    current = select(
        population,
        :year, :age, :pop => :current
    )

    midyear = innerjoin(
        previous, current, on=[:year, :age]
    )
    midyear.population =
        (midyear.previous .+ midyear.current) ./ 2

    # -------------------------------------------------------
    # Construct consistent age-specific observations
    # -------------------------------------------------------

    data = DataFrame(
        year=Int[],
        age_group=String[],
        population=Float64[],
        notifications=Int[],
        rate=Float64[],
    )

    for year in years
        pop_year = filter(:year => ==(year), midyear)
        obs_year = filter(:year => ==(year), notifications)

        for (label, lower, upper) in age_groups
            P = sum(
                pop_year.population[
                    (pop_year.age .>= lower) .& (pop_year.age .< upper)
                ]
            )

            if label == "0-14"
                # Prefer combined observations when provided.
                combined = filter(
                    :age_group => ==("0-14"), obs_year
                )

                if nrow(combined) == 1
                    Y = only(combined.cases)
                else
                    young = filter(
                        :age_group => in(["0-4", "5-14"]),
                        obs_year
                    )

                    nrow(young) == 2 ||
                        error("Incomplete childhood data: $year")

                    Y = sum(young.cases)
                end
            else
                obs = filter(
                    :age_group => ==(label), obs_year
                )

                nrow(obs) == 1 ||
                    error("Missing observation: $year, $label")

                Y = only(obs.cases)
            end

            P > 0 || error("Invalid population: $year, $label")

            push!(data, (
                year, label, P, Y, Y / P
            ))
        end
    end

    CSV.write(
        joinpath(tabdir, "decomposition_age_data.csv"),
        data
    )

    # -------------------------------------------------------
    # Annual population, age shares and rates
    # -------------------------------------------------------

    annual = Dict{Int,NamedTuple}()

    for year in years
        d = filter(:year => ==(year), data)

        P = sum(d.population)
        w = d.population ./ P
        r = copy(d.rate)
        N = sum(d.notifications)

        annual[year] = (P=P, w=w, r=r, N=N)
    end

    # -------------------------------------------------------
    # Direct age standardisation (reference year 2023)
    # -------------------------------------------------------

    reference_weights = annual[2023].w

    rates = DataFrame(
        year=Int[],
        notifications=Int[],
        population=Float64[],
        crude_rate=Float64[],
        standardised_rate=Float64[],
    )

    for year in years
        d = annual[year]

        push!(rates, (
            year,
            d.N,
            d.P,
            1e5 * d.N / d.P,
            1e5 * sum(reference_weights .* d.r),
        ))
    end

    CSV.write(
        joinpath(tabdir, "standardised_notification_rates.csv"),
        rates
    )

    # -------------------------------------------------------
    # Exact three-factor Shapley decomposition
    # -------------------------------------------------------

    orders = (
        (1, 2, 3),
        (1, 3, 2),
        (2, 1, 3),
        (2, 3, 1),
        (3, 1, 2),
        (3, 2, 1),
    )

    function shapley(base, target)

        # Factors:
        # 1 = population size
        # 2 = population age shares
        # 3 = age-specific notification rates

        function evaluate(mask)
            P = mask[1] ? target.P : base.P
            w = mask[2] ? target.w : base.w
            r = mask[3] ? target.r : base.r

            return P * sum(w .* r)
        end

        contributions = zeros(3)

        for order in orders
            mask = falses(3)
            previous_value = evaluate(mask)

            for factor in order
                mask[factor] = true
                current_value = evaluate(mask)

                contributions[factor] +=
                    current_value - previous_value

                previous_value = current_value
            end
        end

        contributions ./= length(orders)

        return contributions
    end

    function decomposition(reference_year)
        output = DataFrame(
            reference_year=Int[],
            year=Int[],
            reference_notifications=Int[],
            notifications=Int[],
            total_change=Float64[],
            size_effect=Float64[],
            ageing_effect=Float64[],
            rate_effect=Float64[],
        )

        base = annual[reference_year]

        for year in years
            target = annual[year]
            effects = shapley(base, target)

            change = target.N - base.N

            isapprox(
                sum(effects), change;
                atol=1e-8, rtol=1e-10
            ) || error("Shapley decomposition failed: $year")

            push!(output, (
                reference_year,
                year,
                base.N,
                target.N,
                change,
                effects[1],
                effects[2],
                effects[3],
            ))
        end

        return output
    end

    long_term = decomposition(2013)
    recent = decomposition(2023)

    CSV.write(
        joinpath(tabdir, "decomposition_from_2013.csv"),
        long_term
    )

    CSV.write(
        joinpath(tabdir, "decomposition_from_2023.csv"),
        recent
    )

    # -------------------------------------------------------
    # Figure 12 — Crude and standardised rates
    # -------------------------------------------------------

    fig = Figure(size=(1000, 550))

    ax = Axis(
        fig[1, 1],
        title="TB notification rates",
        xlabel="Year",
        ylabel="Notifications per 100,000",
    )

    lines!(ax, rates.year, rates.crude_rate;
        linewidth=2, label="Crude")

    lines!(ax, rates.year, rates.standardised_rate;
        linewidth=2, label="Age-standardised (2023)")

    axislegend(ax)
    save(joinpath(figdir, "12_standardised_rates.png"), fig)

    # -------------------------------------------------------
    # Figure 13 — Long-term decomposition
    # -------------------------------------------------------

    fig = Figure(size=(1000, 650))

    ax = Axis(
        fig[1, 1],
        title="Change in notifications relative to 2013",
        xlabel="Year",
        ylabel="Contribution to change in notifications",
    )

    for (field, label) in (
        (:size_effect, "Population size"),
        (:ageing_effect, "Age structure"),
        (:rate_effect, "Age-specific rates"),
    )
        lines!(ax, long_term.year, long_term[!, field];
            linewidth=2, label=label)
    end

    lines!(ax, long_term.year, long_term.total_change;
        linewidth=2, linestyle=:dash,
        color=:black, label="Total change")

    hlines!(ax, [0], color=(:grey, 0.5))
    axislegend(ax, position=:lt)

    save(joinpath(figdir, "13_decomposition_2013.png"), fig)

    # -------------------------------------------------------
    # Figure 14 — Recent decomposition
    # -------------------------------------------------------

    recent_period = filter(:year => >=(2023), recent)

    fig = Figure(size=(900, 550))

    ax = Axis(
        fig[1, 1],
        title="Change in notifications relative to 2023",
        xlabel="Year",
        ylabel="Contribution to change in notifications",
        xticks=([2023, 2024, 2025],
            ["2023", "2024", "2025"]),
    )

    for (field, label) in (
        (:size_effect, "Population size"),
        (:ageing_effect, "Age structure"),
        (:rate_effect, "Age-specific rates"),
    )
        lines!(ax,
            recent_period.year,
            recent_period[!, field];
            linewidth=2, label=label)
    end

    lines!(ax,
        recent_period.year,
        recent_period.total_change;
        linewidth=2, linestyle=:dash,
        color=:black, label="Total change")

    hlines!(ax, [0], color=(:grey, 0.5))
    axislegend(ax)

    save(joinpath(figdir, "14_decomposition_2023.png"), fig)

    # -------------------------------------------------------
    # Results
    # -------------------------------------------------------

    println("\nCRUDE AND STANDARDISED RATES")
    println(rates)

    println("\nDECOMPOSITION RELATIVE TO 2013")
    println(long_term)

    println("\nDECOMPOSITION RELATIVE TO 2023")
    println(recent_period)

    println("\nOutputs saved to: ", outdir)
end
