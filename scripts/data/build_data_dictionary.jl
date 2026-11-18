
using CSV
using DataFrames

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    datadir = joinpath(root, "data", "processed")
    outfile = joinpath(root, "docs", "data", "data_dictionary.md")

    mkpath(dirname(outfile))

    # Dataset-level scientific descriptions.
    descriptions = Dict(
        "tb_notifications_total" => "Reported national new and recurrent TB notifications.",
        "tb_notifications_by_age" => "Reported notifications by age. Childhood ages are combined in 2024.",
        "tb_recurrent_notifications" => "Reported recurrent TB notifications, using historical WHO definitions.",
        "tb_notifications_by_site" => "Reported notifications by pulmonary diagnostic category and extrapulmonary disease.",
        "tb_historical_new_notifications_by_age_and_type" => "Historical smear-positive, smear-negative and extrapulmonary notifications.",
        "tb_new_notifications_by_site_age_2007_2012" => "Pulmonary and extrapulmonary notifications by age and year.",
        "tb_pulmonary_fraction_by_age_2007_2012" => "Pooled observed pulmonary notification proportions.",
        "tb_treatment_outcomes" => "Treatment outcomes for all available WHO cohorts.",
        "tb_treatment_outcomes_main" => "Selected national treatment cohort. Reporting changes in 2024.",
        "tb_treatment_outcomes_age_sex" => "Reported age- and sex-specific treatment outcomes.",
        "tb_burden_national" => "WHO-estimated national TB burden with uncertainty bounds.",
        "tb_incidence_age_sex" => "WHO-estimated age- and sex-specific incidence, 2025.",
        "tb_preventive_treatment" => "WHO preventive-treatment and household-contact estimates.",
        "population" => "WPP2024 population by single-year age, year and sex.",
        "fertility" => "WPP2024 fertility distribution, TFR and derived ASFR.",
        "mortality" => "WPP2024 annual central mortality rates by age.",
        "migration" => "WPP2024 annual national net migration.",
        "vital_totals" => "WPP2024 births, deaths and population-change summaries.",
        "population_totals" => "Annual population totals derived from single-year ages.",
        "fertility_checks" => "Annual checks of age-specific fertility reconstruction.",
        "contact_matrix" => "Original Prem contacts expanded to single-year ages.",
        "prem_reciprocal_2023_96" => "Prem contacts corrected for Kiribati 2023 population reciprocity.",
        "prem_reciprocal_2023_16" => "16-band Prem contacts corrected for 2023 reciprocity.",
    )

    # Column definitions shared across datasets.
    definitions = Dict(
        "year" => ("calendar year", "WHO or WPP reference year"),
        "age" => ("years", "Single-year age"),
        "age_group" => ("age band", "Reported age category"),
        "cases" => ("persons", "Reported TB notifications"),
        "cohort_size" => ("persons", "Treatment cohort denominator"),
        "successful" => ("persons", "Successful treatment outcomes"),
        "died" => ("persons", "Recorded treatment deaths"),
        "failed" => ("persons", "Treatment failures"),
        "lost_to_followup" => ("persons", "Lost to follow-up"),
        "not_evaluated" => ("persons", "Treatment outcome not evaluated"),
        "who_cohort" => ("category", "WHO treatment cohort definition"),
        "death_measure" => ("category", "Definition of recorded treatment deaths"),
        "pulmonary" => ("persons", "Pulmonary notifications"),
        "extrapulmonary" => ("persons", "Extrapulmonary notifications"),
        "pulmonary_fraction" => ("proportion", "Pulmonary notifications divided by total"),
        "pop" => ("persons", "Total population"),
        "popM" => ("persons", "Male population"),
        "popF" => ("persons", "Female population"),
        "population" => ("persons", "Population count"),
        "tfr" => ("births/woman", "Total fertility rate"),
        "pasfr" => ("percent", "Age-specific share of total fertility"),
        "asfr" => ("births/woman/year", "Age-specific fertility rate"),
        "mxB" => ("1/year", "Both-sex central mortality rate"),
        "migration" => ("persons/year", "Annual net migration"),
        "mig" => ("thousands/year", "Original WPP migration value"),
        "e_inc_num" => ("persons/year", "WHO estimated annual incident TB cases"),
        "e_mort_num" => ("persons/year", "WHO estimated annual TB deaths"),
        "c_cdr" => ("percent", "WHO estimated case detection rate"),
    )

    # Dataset discovery, including nested estimates.
    paths = String[]
    for (dir, _, files) in walkdir(datadir)
        for file in files
            endswith(lowercase(file), ".csv") || continue
            push!(paths, joinpath(dir, file))
        end
    end
    sort!(paths)

    open(outfile, "w") do io
        println(io, "# Processed data dictionary\n")
        println(io,
            "Generated from the current processed CSV schemas. " *
                "Scientific definitions should be reviewed when sources change.\n"
        )

        for path in paths
            name = splitext(basename(path))[1]
            relative = replace(relpath(path, root), '\\' => '/')

            df = CSV.read(path, DataFrame)

            println(io, "## `", relative, "`\n")
            println(io, get(descriptions, name,
                "Validation or derived dataset; definition requires review."))
            println(io)
            println(io, "Records: **", nrow(df), "**\n")
            println(io, "| Variable | Type | Unit | Definition |")
            println(io, "|---|---|---|---|")

            for col in names(df)
                unit, meaning = get(
                    definitions, col,
                    ("—", "Source field; definition requires review")
                )

                type = string(eltype(df[!, col]))

                println(io,
                    "| `", col, "` | `", type,
                    "` | ", unit, " | ", meaning, " |"
                )
            end

            println(io)
        end
    end

    println("Data dictionary generated: ", outfile)
    println("Datasets documented: ", length(paths))
end
