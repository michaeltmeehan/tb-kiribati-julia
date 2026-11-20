
# src/data/surveillance.jl

using CSV

# ------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------

function _who_rows(path::AbstractString, columns::Tuple; years=nothing)
    rows = collect(CSV.File(path))

    isempty(rows) &&
        throw(ArgumentError("Empty WHO data file: $path"))

    available = propertynames(first(rows))
    for column in columns
        column in available ||
            throw(ArgumentError("Missing column $column in $path"))
    end

    if !isnothing(years)
        selected = Set(Int(y) for y in years)
        filter!(r -> Int(r.year) in selected, rows)
    end

    return rows
end

function _who_count(value, description; allow_missing=false)
    if ismissing(value)
        allow_missing ||
            throw(ArgumentError("Missing count: $description"))
        return missing
    end

    count = Int(value)
    count >= 0 ||
        throw(ArgumentError("Negative count: $description"))

    return count
end

function _who_unique!(rows, key)
    seen = Set{Any}()

    for row in rows
        k = key(row)
        k in seen &&
            throw(ArgumentError("Duplicate WHO record: $k"))
        push!(seen, k)
    end

    return rows
end


# ------------------------------------------------------------
# National counts: total or recurrent notifications
# ------------------------------------------------------------

"""
    load_annual_counts(path; years=nothing)

Read year,cases observations.

Missing years remain absent rather than being assigned zero.
"""
function load_annual_counts(path::AbstractString; years=nothing)
    source = _who_rows(path, (:year, :cases); years)

    rows = [
        (
            year=Int(r.year),
            cases=_who_count(r.cases, "year $(r.year)"),
        )
        for r in source
    ]

    _who_unique!(rows, r -> r.year)
    sort!(rows; by=r -> r.year)

    return rows
end


# ------------------------------------------------------------
# Notifications by disease site
# ------------------------------------------------------------

"""
    load_site_notifications(path; years=nothing)

Read national pulmonary and extrapulmonary notifications.
Pulmonary counts are calculated from the two reported
pulmonary categories.
"""
function load_site_notifications(path::AbstractString; years=nothing)
    columns = (
        :year,
        :pulmonary_lab_confirmed,
        :pulmonary_clinically_diagnosed,
        :extrapulmonary,
        :total,
    )

    source = _who_rows(path, columns; years)
    rows = NamedTuple[]

    for r in source
        lab = _who_count(r.pulmonary_lab_confirmed, "pulmonary lab")
        clin = _who_count(r.pulmonary_clinically_diagnosed, "pulmonary clinical")
        ep = _who_count(r.extrapulmonary, "extrapulmonary")
        total = _who_count(r.total, "total")

        pulmonary = lab + clin

        pulmonary + ep == total ||
            throw(ArgumentError(
                "Inconsistent site totals in $(r.year)"
            ))

        push!(rows, (
            year=Int(r.year),
            pulmonary_lab_confirmed=lab,
            pulmonary_clinically_diagnosed=clin,
            pulmonary=pulmonary,
            extrapulmonary=ep,
            total=total,
        ))
    end

    _who_unique!(rows, r -> r.year)
    sort!(rows; by=r -> r.year)

    return rows
end


# ------------------------------------------------------------
# Treatment outcomes
# ------------------------------------------------------------

"""
    load_treatment_outcomes(path; years=nothing)

Read either the main or historical treatment-outcome CSV.

Retains WHO cohort categories and death definitions.
Missing outcome counts remain missing, not zero.
"""
function load_treatment_outcomes(path::AbstractString; years=nothing)
    columns = (
        :year,
        :who_cohort,
        :cohort_size,
        :successful,
        :died,
        :failed,
        :lost_to_followup,
        :not_evaluated,
        :death_measure,
    )

    source = _who_rows(path, columns; years)
    rows = NamedTuple[]

    for r in source
        year = Int(r.year)
        cohort = String(r.who_cohort)

        count(field) = _who_count(
            getproperty(r, field),
            "$year/$cohort/$field";
            allow_missing=true,
        )

        record = (
            year=year,
            who_cohort=cohort,
            cohort_size=_who_count(
                r.cohort_size, "$year/$cohort/cohort_size"
            ),
            successful=count(:successful),
            died=count(:died),
            failed=count(:failed),
            lost_to_followup=count(:lost_to_followup),
            not_evaluated=count(:not_evaluated),
            death_measure=String(r.death_measure),
        )

        outcomes = (
            record.successful,
            record.died,
            record.failed,
            record.lost_to_followup,
            record.not_evaluated,
        )

        # Validate only when every outcome is reported.
        if all(x -> !ismissing(x), outcomes)
            sum(outcomes) == record.cohort_size ||
                throw(ArgumentError(
                    "Treatment outcomes do not sum to cohort size: " *
                        "$year/$cohort"
                ))
        end

        push!(rows, record)
    end

    _who_unique!(rows, r -> (r.year, r.who_cohort))
    sort!(rows; by=r -> (r.year, r.who_cohort))

    return rows
end


# ------------------------------------------------------------
# Historical notifications by age and diagnostic category
# ------------------------------------------------------------

"""
    load_historical_notification_types(path; years=nothing)

Read year,age_group,who_category,cases.

Retains partial historical disaggregations without
imputing unreported categories or ages.
"""
function load_historical_notification_types(
    path::AbstractString;
    years=nothing,
)
    source = _who_rows(
        path,
        (:year, :age_group, :who_category, :cases);
        years,
    )

    rows = [
        (
            year=Int(r.year),
            age_group=String(r.age_group),
            who_category=String(r.who_category),
            cases=_who_count(
                r.cases,
                "$(r.year)/$(r.age_group)/$(r.who_category)",
            ),
        )
        for r in source
    ]

    _who_unique!(
        rows,
        r -> (r.year, r.age_group, r.who_category),
    )

    sort!(rows; by=r -> (
        r.year, r.who_category, r.age_group
    ))

    return rows
end


# ------------------------------------------------------------
# Convenience loader for all Kiribati surveillance data
# ------------------------------------------------------------

"""
    load_kiribati_surveillance(data_dir;
        age_labels,
        age_years=2013:2025,
    )

Load all eight cleaned WHO datasets.

Age-specific observations use the existing notification
loader and its model-age-group mappings.
"""
function load_kiribati_surveillance(
    data_dir::AbstractString;
    age_labels,
    age_years=2013:2025,
)
    file(name) = joinpath(data_dir, name)

    return (
        age_notifications=load_notification_observations(
            file("tb_notifications_by_age.csv");
            age_labels,
            years=age_years,
        ),
        national_notifications=load_annual_counts(
            file("tb_notifications_total.csv"),
        ),
        recurrent_notifications=load_annual_counts(
            file("tb_recurrent_notifications.csv"),
        ),
        site_notifications=load_site_notifications(
            file("tb_notifications_by_site.csv"),
        ),
        treatment_outcomes=load_treatment_outcomes(
            file("tb_treatment_outcomes_main.csv"),
        ),
        treatment_outcomes_all=load_treatment_outcomes(
            file("tb_treatment_outcomes.csv"),
        ),
        historical_new_by_type=load_historical_notification_types(
            file("tb_historical_new_notifications_by_age_and_type.csv"),
        ),
    )
end



"""
    combine_notification_observations(
        age_observations,
        national_notifications;
        n_age_groups,
        start_year,
        end_year,
        national_start_year=1980,
    )

Use age-specific observations whenever available;
otherwise use the reported national total.

Exactly one observation record is retained per year.
No missing years or age counts are imputed.
"""
function combine_notification_observations(
    age_observations,
    national_notifications;
    n_age_groups::Integer,
    start_year::Integer,
    end_year::Integer,
    national_start_year::Integer=1980,
)
    n_age_groups > 0 ||
        throw(ArgumentError("n_age_groups must be positive"))

    start_year < end_year ||
        throw(ArgumentError("Invalid simulation interval"))

    combined = NamedTuple[]
    age_years = Set{Int}()

    # Age-specific observations take precedence.
    for obs in age_observations
        year = Int(obs.year)

        start_year <= year < end_year || continue

        year in age_years &&
            throw(ArgumentError("Duplicate age year $year"))

        push!(age_years, year)
        push!(combined, obs)
    end

    # Use national totals only where age-specific
    # observations are not already included.
    national_years = Set{Int}()

    for row in national_notifications
        year = Int(row.year)

        year in national_years &&
            throw(ArgumentError("Duplicate national year $year"))

        push!(national_years, year)

        max(start_year, national_start_year) <= year < end_year ||
            continue

        year in age_years && continue

        count = Int(row.cases)
        count >= 0 ||
            throw(ArgumentError("Negative count in $year"))

        push!(combined, (
            year=year,
            labels=["all"],
            counts=[count],
            groups=[collect(1:Int(n_age_groups))],
        ))
    end

    sort!(combined; by=obs -> obs.year)

    isempty(combined) &&
        throw(ArgumentError("No notification observations"))

    return combined
end
