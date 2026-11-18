
using CSV
using DataFrames
using CairoMakie
using Statistics

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    dir = joinpath(root, "data", "processed", "surveillance")
    estimates_dir = joinpath(dir, "estimates")
    outdir = joinpath(root, "results", "data_exploration")
    figdir = joinpath(outdir, "figures")
    tabdir = joinpath(outdir, "tables")

    mkpath(figdir)
    mkpath(tabdir)

    load(name) = CSV.read(joinpath(dir, "$name.csv"), DataFrame)
    load_est(name) = CSV.read(
        joinpath(estimates_dir, "$name.csv"), DataFrame
    )

    export_table(name, df) =
        CSV.write(joinpath(tabdir, "$name.csv"), df)

    export_figure(name, fig) =
        save(joinpath(figdir, "$name.png"), fig)

    notifications = load("tb_notifications_total")
    ages = load("tb_notifications_by_age")
    sites = load("tb_notifications_by_site")
    recurrence = load("tb_recurrent_notifications")
    treatment = load("tb_treatment_outcomes_main")
    pulmonary = load("tb_pulmonary_fraction_by_age_2007_2012")
    burden = load_est("tb_burden_national")

    age_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+"
    ]

    # ---------------------------------------------------------
    # 1. Notifications vs estimated incidence
    # ---------------------------------------------------------

    fig = Figure(size=(1000, 550))
    ax = Axis(fig[1, 1],
        title="Notifications and WHO-estimated incidence",
        xlabel="Year", ylabel="Annual cases")

    lines!(ax, notifications.year, notifications.cases,
        label="Notifications", linewidth=2)

    inc = dropmissing(select(
        burden, :year, :e_inc_num,
        :e_inc_num_lo, :e_inc_num_hi
    ))

    band!(ax, inc.year,
        inc.e_inc_num_lo, inc.e_inc_num_hi,
        color=(:grey, 0.25))

    lines!(ax, inc.year, inc.e_inc_num,
        color=:black, linewidth=2,
        label="WHO estimated incidence")

    axislegend(ax, position=:lt)
    export_figure("01_notifications_incidence", fig)

    # ---------------------------------------------------------
    # 2. Age-specific notifications
    # ---------------------------------------------------------

    years = sort(unique(ages.year))
    matrix = fill(NaN, length(age_labels), length(years))

    for r in eachrow(ages)
        i = findfirst(==(r.age_group), age_labels)
        j = findfirst(==(r.year), years)

        # Combined childhood groups are not split.
        if !isnothing(i) && !isnothing(j)
            matrix[i, j] = r.cases
        end
    end


    fig = Figure(size=(1000, 750))

    panels = [
        age_labels[1:4],
        age_labels[5:8],
    ]

    for (i, labels) in enumerate(panels)
        ax = Axis(
            fig[i, 1],
            title=i == 1 ? "Ages 0–34" : "Ages 35+",
            xlabel=i == 2 ? "Year" : "",
            ylabel="Notifications",
        )

        for (j, label) in enumerate(labels)
            d = filter(:age_group => ==(label), ages)
            sort!(d, :year)

            complete_years = collect(minimum(d.year):maximum(d.year))

            counts = [
                y in d.year ? d.cases[findfirst(==(y), d.year)] : NaN
                for y in complete_years
            ]

            lines!(ax, complete_years, counts;
                color=Cycled(j), linewidth=2,
                label=label
            )
            scatter!(ax, d.year, d.cases;
                color=Cycled(j), markersize=7
            )
        end

        axislegend(ax, position=:rt)
    end

    export_figure("02_notifications_by_age", fig)


    # ---------------------------------------------------------
    # 3. Disease presentation
    # ---------------------------------------------------------


    sites.pulmonary =
        sites.pulmonary_lab_confirmed .+
            sites.pulmonary_clinically_diagnosed

    sites.lab_fraction =
        sites.pulmonary_lab_confirmed ./ sites.pulmonary

    fig = Figure(size=(1000, 750))

    ax1 = Axis(fig[1, 1],
        title="TB notifications by presentation",
        ylabel="Annual notifications")

    for (j, (field, label)) in enumerate([
        (:pulmonary, "Total pulmonary"),
        (:pulmonary_lab_confirmed, "Lab confirmed"),
        (:pulmonary_clinically_diagnosed, "Clinically diagnosed"),
        (:extrapulmonary, "Extrapulmonary"),
    ])
        lines!(ax1, sites.year, sites[!, field];
            color=Cycled(j), linewidth=2, label=label)
    end

    axislegend(ax1, position=:lt)

    ax2 = Axis(fig[2, 1],
        xlabel="Year",
        ylabel="Lab-confirmed fraction of pulmonary TB",
        limits=(nothing, (0, 1)))

    lines!(ax2, sites.year, sites.lab_fraction;
        linewidth=2)
    scatter!(ax2, sites.year, sites.lab_fraction)

    export_table("disease_presentation_summary", sites)
    export_figure("03_disease_presentation", fig)


    # ---------------------------------------------------------
    # 4. Recurrent notifications
    # ---------------------------------------------------------

    rec = innerjoin(
        rename(copy(recurrence), :cases => :recurrent),
        rename(copy(notifications), :cases => :total),
        on=:year
    )

    rec.recurrent_fraction =
        rec.recurrent ./ rec.total

    export_table("recurrence_summary", rec)

    fig = Figure(size=(1000, 650))
    ax1 = Axis(fig[1, 1],
        title="Recurrent notifications",
        ylabel="Notifications")

    lines!(ax1, rec.year, rec.recurrent, linewidth=2)

    ax2 = Axis(fig[2, 1],
        xlabel="Year", ylabel="Percent recurrent")

    lines!(ax2, rec.year,
        100 .* rec.recurrent_fraction, linewidth=2)

    export_figure("04_recurrence", fig)

    # ---------------------------------------------------------
    # 5. Treatment outcomes
    # ---------------------------------------------------------

    tx = copy(treatment)

    for field in (
        :successful, :died, :failed,
        :lost_to_followup, :not_evaluated
    )
        newname = Symbol(string(field) * "_fraction")
        tx[!, newname] =
            tx[!, field] ./ tx.cohort_size
    end

    export_table("treatment_summary", tx)

    fig = Figure(size=(1000, 550))
    ax = Axis(fig[1, 1],
        title="Treatment outcomes",
        xlabel="Cohort year", ylabel="Percent of cohort")

    for (i, (field, label)) in enumerate([
        (:successful_fraction, "Successful"),
        (:died_fraction, "Died"),
        (:lost_to_followup_fraction, "Lost"),
        (:failed_fraction, "Failed"),
        (:not_evaluated_fraction, "Not evaluated")
    ])
        valid = dropmissing(select(tx, :year, field))
        lines!(ax, valid.year, 100 .* valid[!, field],
            color=Cycled(i), linewidth=2, label=label)
    end

    axislegend(ax, position=:rc)
    export_figure("05_treatment_outcomes", fig)

    # ---------------------------------------------------------
    # 6. WHO burden estimates
    # ---------------------------------------------------------

    fig = Figure(size=(1000, 750))

    for (row, (best, lo, hi, label)) in enumerate([
        (:e_inc_num, :e_inc_num_lo, :e_inc_num_hi,
            "Estimated incident TB cases"),
        (:e_mort_num, :e_mort_num_lo, :e_mort_num_hi,
            "Estimated TB deaths"),
        (:c_cdr, :c_cdr_lo, :c_cdr_hi,
            "Estimated case detection (%)")
    ])
        ax = Axis(fig[row, 1],
            xlabel=row == 3 ? "Year" : "",
            ylabel=label)

        d = dropmissing(select(burden, :year, best, lo, hi))

        band!(ax, d.year, d[!, lo], d[!, hi],
            color=(:grey, 0.25))

        lines!(ax, d.year, d[!, best],
            color=:black, linewidth=2)
    end

    export_figure("06_who_burden", fig)

    # ---------------------------------------------------------
    # 7. Historical pulmonary notification fractions
    # ---------------------------------------------------------


    # Wilson 95% binomial interval
    function wilson_interval(k, n; z=1.96)
        p = k / n
        den = 1 + z^2 / n
        centre = (p + z^2 / (2n)) / den
        half = z / den *
            sqrt(p * (1-p) / n + z^2 / (4n^2))
        return centre - half, centre + half
    end

    historical = load(
        "tb_new_notifications_by_site_age_2007_2012"
    )

    fig = Figure(size=(1000, 500))
    ax = Axis(fig[1, 1],
        title="Pulmonary notification fractions, 2007–2012",
        xlabel="Age group",
        ylabel="Pulmonary fraction",
        xticks=(1:8, age_labels),
        limits=(nothing, (0, 1)))

    for (i, age) in enumerate(age_labels)
        d = filter(:age_group => ==(age), historical)

        # Individual annual proportions
        annual = d.pulmonary ./ d.total

        scatter!(ax,
            fill(i, length(annual)), annual;
            color=(:grey, 0.45),
            markersize=7
        )

        # Pooled estimate and Wilson interval
        k = sum(d.pulmonary)
        n = sum(d.total)
        p = k / n
        lo, hi = wilson_interval(k, n)

        errorbars!(ax, [i], [p],
            [p-lo], [hi-p];
            whiskerwidth=12,
            color=:black
        )

        scatter!(ax, [i], [p];
            color=:black, markersize=11)
    end

    export_figure("07_historical_pulmonary_fraction", fig)


    # ---------------------------------------------------------
    # 8. Consistency and coverage reports
    # ---------------------------------------------------------

    checks = [
        "notification_checks",
        "site_checks",
        "treatment_checks",
        "historical_presentation_checks"
    ]

    for name in checks
        src = joinpath(dir, "validation", "$name.csv")
        if isfile(src)
            df = CSV.read(src, DataFrame)
            export_table(name, df)
        end
    end

    datasets = (
        notifications=notifications,
        age_notifications=ages,
        disease_site=sites,
        recurrence=recurrence,
        treatment=treatment,
        who_burden=burden
    )

    coverage = DataFrame(
        dataset=String[],
        first_year=Int[],
        last_year=Int[],
        records=Int[]
    )

    for (name, df) in pairs(datasets)
        push!(coverage, (
            string(name),
            minimum(df.year),
            maximum(df.year),
            nrow(df)
        ))
    end

    export_table("data_coverage", coverage)

    println("\nDATA COVERAGE")
    println(coverage)
    println("\nFigures saved to: ", figdir)
    println("Tables saved to: ", tabdir)
end
