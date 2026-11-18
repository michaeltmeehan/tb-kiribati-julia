
using CSV
using DataFrames
using Printf

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    results = joinpath(root, "results", "data_exploration")
    tables = joinpath(results, "tables")
    surveillance = joinpath(
        root, "data", "processed", "surveillance"
    )
    demography = joinpath(
        root, "data", "processed", "demography"
    )

    output = joinpath(root, "docs", "data", "exploration.md")
    mkpath(dirname(output))

    readtable(dir, name) =
        CSV.read(joinpath(dir, "$name.csv"), DataFrame)

    notifications = readtable(
        surveillance, "tb_notifications_total"
    )
    sites = readtable(
        surveillance, "tb_notifications_by_site"
    )
    recurrence = readtable(tables, "recurrence_summary")
    rates = readtable(
        tables, "standardised_notification_rates"
    )
    decomposition = readtable(
        tables, "decomposition_from_2013"
    )
    recent = readtable(
        tables, "decomposition_from_2023"
    )
    pop = readtable(demography, "population_totals")

    burden = readtable(
        joinpath(surveillance, "estimates"),
        "tb_burden_national"
    )

    # Convenience functions.
    record(df, year) = only(
        eachrow(filter(:year => ==(year), df))
    )

    fmt(x; digits=1) =
        ismissing(x) ? "NA" :
        @sprintf("%.*f", digits, Float64(x))

    function markdown_table(io, headers, rows)
        println(io, "| ", join(headers, " | "), " |")
        println(io, "| ", join(fill("---", length(headers)), " | "), " |")

        for row in rows
            println(io, "| ", join(string.(row), " | "), " |")
        end
        println(io)
    end

    open(output, "w") do io
        println(io, "# Exploratory analysis of Kiribati TB data\n")

        println(
            io,
            """
## Scope

This document summarises the exploratory analyses of WHO Global
TB Report 2026 surveillance data and WPP2024 demographic data.

All numerical results are generated from processed CSVs.
Figures are available under `results/data_exploration/figures/`.

Observed notifications and WHO burden estimates are distinct
quantities and must not be interpreted interchangeably.
"""
        )

        # --------------------------------------------------
        # Notifications
        # --------------------------------------------------

        println(io, "\n## National notifications\n")

        markdown_table(io,
            ["Year", "Notifications"],
            [
                [y, record(notifications, y).cases]
                for y in (2013, 2020, 2023, 2024, 2025)
            ]
        )

        println(
            io,
            """
Notification counts reflect both underlying TB disease
occurrence and ascertainment, including diagnostic and
reporting practices.

Age-specific notification counts are available for 2013–2025.
In 2024, childhood observations are reported as a combined
0–14 age group.
"""
        )

        # --------------------------------------------------
        # Disease presentation
        # --------------------------------------------------

        println(io, "\n## Disease presentation\n")

        markdown_table(io,
            [
                "Year", "Laboratory-confirmed pulmonary",
                "Clinically diagnosed pulmonary",
                "Extrapulmonary"
            ],
            [
                begin
                    r = record(sites, y)
                    [
                        y,
                        r.pulmonary_lab_confirmed,
                        r.pulmonary_clinically_diagnosed,
                        r.extrapulmonary
                    ]
                end
                for y in (2013, 2020, 2025)
            ]
        )

        println(
            io,
            """
Pulmonary notifications comprise laboratory-confirmed and
clinically diagnosed pulmonary notifications.

Changes in these categories may reflect diagnostic or
reporting practices, not solely changes in disease burden.

Historical smear-positive and smear-negative classifications
are not directly equivalent to modern diagnostic categories.

The historical 2007–2012 pulmonary fractions describe
observed notifications, not underlying disease-onset
probabilities. Their figures show pooled Wilson intervals
and individual annual proportions.
"""
        )

        # --------------------------------------------------
        # Recurrence
        # --------------------------------------------------

        println(io, "\n## Recurrent notifications\n")

        markdown_table(io,
            ["Year", "Recurrent", "Total", "Percent recurrent"],
            [
                begin
                    r = record(recurrence, y)
                    [
                        y, r.recurrent, r.total,
                        fmt(100*r.recurrent_fraction)
                    ]
                end
                for y in (2013, 2020, 2024)
            ]
        )

        println(
            io,
            """
Recurrent notification definitions changed historically.
Trends should be interpreted in the context of those changes.

The recurrent proportion is conditional on reported total
notifications, rather than an independent incidence measure.
"""
        )

        # --------------------------------------------------
        # WHO burden
        # --------------------------------------------------

        println(io, "\n## WHO national burden estimates\n")

        markdown_table(io,
            [
                "Year", "Incident cases",
                "Incident interval", "TB deaths",
                "Case detection (%)"
            ],
            [
                begin
                    r = record(burden, y)
                    [
                        y,
                        r.e_inc_num,
                        "$(r.e_inc_num_lo)–$(r.e_inc_num_hi)",
                        r.e_mort_num,
                        fmt(r.c_cdr)
                    ]
                end
                for y in (2013, 2020, 2023, 2024, 2025)
            ]
        )

        println(
            io,
            """
WHO incidence, mortality and case-detection estimates
are model-derived quantities, with reported uncertainty.

WHO case-detection estimates are not independent of
notifications and cannot automatically be added to a
notification likelihood as independent observations.

Age-specific WHO incidence estimates are not treated as
independent age-specific observations in this project.
"""
        )

        # --------------------------------------------------
        # Demography
        # --------------------------------------------------

        println(io, "\n## Demographic trends\n")

        markdown_table(io,
            ["Year", "Population"],
            [
                [y, round(Int, record(pop, y).population)]
                for y in (1950, 2000, 2023, 2025, 2050, 2100)
            ]
        )

        println(
            io,
            """
Demographic estimates and projections come from WPP2024.
Population values represent end-of-year populations.

Annual notification rates use the average of consecutive
end-of-year populations to approximate midyear person-time.
This approximation should be reconsidered if exact
exposure estimates become available.
"""
        )

        # --------------------------------------------------
        # Standardised rates
        # --------------------------------------------------

        println(io, "\n## Notification rates\n")

        markdown_table(io,
            [
                "Year", "Crude rate",
                "Age-standardised rate"
            ],
            [
                begin
                    r = record(rates, y)
                    [
                        y,
                        fmt(r.crude_rate),
                        fmt(r.standardised_rate)
                    ]
                end
                for y in (2013, 2020, 2023, 2024, 2025)
            ]
        )

        println(
            io,
            """
Rates are per 100,000 population.

Direct age standardisation uses Kiribati's 2023
population distribution as the reference.

Seven consistent age groups are used:
0–14, 15–24, 25–34, 35–44, 45–54, 55–64 and 65+.
This avoids artificially splitting the combined
childhood observations in 2024.
"""
        )

        # --------------------------------------------------
        # Shapley decomposition
        # --------------------------------------------------

        println(io, "\n## Demographic decomposition\n")

        println(
            io,
            """
Notification changes are decomposed into contributions from:

1. Total population size.
2. Population age composition.
3. Age-specific notification rates.

The decomposition averages marginal contributions across
all six factor orderings (three-factor Shapley method).

The contributions sum exactly to the observed change
in age-aggregated notifications.
"""
        )

        function decomposition_row(df, year)
            r = record(df, year)
            return [
                "$(r.reference_year)–$(r.year)",
                fmt(r.size_effect),
                fmt(r.ageing_effect),
                fmt(r.rate_effect),
                fmt(r.total_change)
            ]
        end

        markdown_table(io,
            [
                "Period", "Size", "Age structure",
                "Rates", "Total"
            ],
            [
                decomposition_row(decomposition, 2025),
                decomposition_row(recent, 2025)
            ]
        )

        println(
            io,
            """
The rate contribution includes changes in underlying
disease incidence, case detection and reporting.

The decomposition uses age-specific notification totals.
These differ from independently reported national totals
in 2013 and 2015.
"""
        )

        # --------------------------------------------------
        # Limitations and reproducibility
        # --------------------------------------------------

        println(io, "\n## Limitations and outstanding questions\n")

        println(
            io,
            """
- Surveillance definitions and reporting practices change.
- WHO burden estimates depend partly on reported notifications.
- Treatment-cohort mortality differs from population-level TB mortality.
- The Solomon Islands contact matrix is a proxy for Kiribati.
- The 2024 childhood observations cannot be disaggregated.
- Age-specific notification-rate changes cannot be attributed
  exclusively to epidemiological changes.

## Reproducing the analyses

From the Julia project environment, run:

```julia
include("scripts/data/explore_surveillance.jl")
include("scripts/data/explore_demography.jl")
include("scripts/data/decompose_notifications.jl")
include("scripts/data/document_exploration.jl")
```

All analyses consume processed data without modifying them.
"""
        )
    end

    println("Wrote: ", output)
end
