
module WHO2026

using CSV
using DataFrames

export build_surveillance, build_estimates

const AGE_GROUPS = [
    ("0-4", "04"),
    ("5-14", "514"),
    ("0-14", "014"),
    ("15-24", "1524"),
    ("25-34", "2534"),
    ("35-44", "3544"),
    ("45-54", "4554"),
    ("55-64", "5564"),
    ("65+", "65"),
]

const OUTCOME_FIELDS = (
    ("cohort_size", "coh"),
    ("successful", "succ"),
    ("died", "died"),
    ("failed", "fail"),
    ("lost_to_followup", "lost"),
    ("not_evaluated", "neval"),
)

const TREATMENT_COHORTS = (
    "new_sp",
    "new_snep",
    "ret",
    "newrel",
    "ret_nrel",
    "nrr_tx",
)

# Preserve missingness; reject negative/non-integral counts.
function count_or_missing(x)
    ismissing(x) && return missing
    isfinite(x) || error("Non-finite count: $x")
    x >= 0 && isinteger(x) ||
        error("Invalid count: $x")
    return Int(x)
end

function add_counts(xs...)
    any(ismissing, xs) && return missing
    return sum(xs)
end

function kiribati(path)
    df = CSV.read(path, DataFrame)
    :iso3 in propertynames(df) ||
        error("Missing iso3 in $path")

    df = filter(:iso3 => ==("KIR"), df)
    isempty(df) && error("No Kiribati records in $path")
    return df
end

function check_years(df)
    any(ismissing, df.year) &&
        error("Missing observation year")
    allunique(df.year) ||
        error("Duplicate observation years")
    sort!(df, :year)
    return df
end

function national_notifications(raw)
    df = select(raw, :year, :c_newinc => :cases)
    df.cases = count_or_missing.(df.cases)
    filter!(:cases => !ismissing, df)
    return check_years(df)
end

function age_notifications(raw)
    out = DataFrame(
        year=Int[],
        age_group=String[],
        cases=Union{Missing,Int}[],
    )

    for r in eachrow(raw)
        for (label, suffix) in AGE_GROUPS
            male = count_or_missing(
                r[Symbol("newrel_m", suffix)]
            )
            female = count_or_missing(
                r[Symbol("newrel_f", suffix)]
            )
            cases = add_counts(male, female)

            # Retain only reported age groups.
            ismissing(cases) && continue

            push!(out, (
                Int(r.year), label, cases
            ))
        end
    end

    sort!(out, [:year, :age_group])
    return out
end

function recurrent_notifications(raw)
    out = DataFrame(year=Int[], cases=Int[])

    for r in eachrow(raw)
        y = Int(r.year)

        cases = if y <= 2012
            count_or_missing(r.ret_rel)
        else
            add_counts(
                count_or_missing(r.ret_rel_labconf),
                count_or_missing(r.ret_rel_clindx),
                count_or_missing(r.ret_rel_ep),
            )
        end

        ismissing(cases) && continue
        push!(out, (y, cases))
    end

    return check_years(out)
end

function treatment_outcomes(raw)
    out = DataFrame(
        year=Int[],
        who_cohort=String[],
        cohort_size=Int[],
        successful=Union{Missing,Int}[],
        died=Union{Missing,Int}[],
        failed=Union{Missing,Int}[],
        lost_to_followup=Union{Missing,Int}[],
        not_evaluated=Union{Missing,Int}[],
        death_measure=String[],
    )

    for r in eachrow(raw)
        for cohort in TREATMENT_COHORTS
            prefix = cohort * "_"

            size = count_or_missing(
                r[Symbol(prefix * "coh")]
            )
            ismissing(size) && continue

            recent = cohort == "nrr_tx"

            death_field = recent ?
                          "died_after_tx" : "died"

            lost_field = recent ?
                         "lost_after_tx" : "lost"

            death_measure = recent ?
                            "died_after_treatment_start" :
                            "died_during_treatment_course"

            push!(out, (
                year=Int(r.year),
                who_cohort=cohort,
                cohort_size=size,
                successful=count_or_missing(
                    r[Symbol(prefix * "succ")]
                ),
                died=count_or_missing(
                    r[Symbol(prefix * death_field)]
                ),
                failed=count_or_missing(
                    r[Symbol(prefix * "fail")]
                ),
                lost_to_followup=count_or_missing(
                    r[Symbol(prefix * lost_field)]
                ),
                not_evaluated=count_or_missing(
                    r[Symbol(prefix * "neval")]
                ),
                death_measure=death_measure,
            ))
        end
    end

    sort!(out, [:year, :who_cohort])
    return out
end

function main_treatment_outcomes(all_outcomes)
    main = filter(
        r -> (
            (2012 <= r.year <= 2023 &&
                r.who_cohort == "newrel") ||
                (r.year >= 2024 &&
                r.who_cohort == "nrr_tx")
        ),
        all_outcomes,
    )

    select!(main, [
        :year,
        :cohort_size,
        :died,
        :successful,
        :failed,
        :lost_to_followup,
        :not_evaluated,
        :who_cohort,
        :death_measure,
    ])

    return check_years(main)
end

function age_sex_outcomes(raw)
    cols = [
        :year, :cohort_type, :age_group, :sex,
        :coh, :succ, :fail, :died, :lost, :neval,
    ]

    out = select(raw, cols)

    for c in (:coh, :succ, :fail, :died, :lost, :neval)
        out[!, c] = count_or_missing.(out[!, c])
    end

    sort!(out, [:year, :cohort_type, :age_group, :sex])
    return out
end

# Diagnostic report; discrepancies are retained, not corrected.
function notification_checks(national, by_age)
    out = DataFrame(
        year=Int[],
        age_total=Union{Missing,Int}[],
        national_total=Int[],
        difference=Union{Missing,Int}[],
    )

    for r in eachrow(national)
        ages = by_age[by_age.year .== r.year, :]
        isempty(ages) && continue

        lookup = Dict(
            row.age_group => row.cases
            for row in eachrow(ages)
        )

        child =
            if haskey(lookup, "0-4") &&
                haskey(lookup, "5-14")
                lookup["0-4"] + lookup["5-14"]
            else
                get(lookup, "0-14", missing)
            end

        older = [
            get(lookup, label, missing)
            for (label, _) in AGE_GROUPS
            if !(label in ("0-4", "5-14", "0-14"))
        ]

        total = add_counts(child, older...)
        difference = ismissing(total) ?
                     missing : total - r.cases

        push!(out, (
            r.year, total, r.cases, difference
        ))
    end

    return out
end

function treatment_checks(outcomes)
    out = DataFrame(
        year=Int[],
        cohort_size=Int[],
        outcomes_total=Union{Missing,Int}[],
        difference=Union{Missing,Int}[],
    )

    for r in eachrow(outcomes)
        total = add_counts(
            r.successful, r.died, r.failed,
            r.lost_to_followup, r.not_evaluated,
        )
        difference = ismissing(total) ?
                     missing : total - r.cohort_size

        push!(out, (
            r.year, r.cohort_size, total, difference
        ))
    end

    return out
end


function site_notifications(raw)
    out = DataFrame(
        year=Int[],
        pulmonary_lab_confirmed=Int[],
        pulmonary_clinically_diagnosed=Int[],
        extrapulmonary=Int[],
        total=Int[],
    )

    for r in eachrow(raw)
        y = Int(r.year)
        y >= 2013 || continue

        values = if y <= 2024
            (
                add_counts(
                    count_or_missing(r.new_labconf),
                    count_or_missing(r.ret_rel_labconf),
                ),
                add_counts(
                    count_or_missing(r.new_clindx),
                    count_or_missing(r.ret_rel_clindx),
                ),
                add_counts(
                    count_or_missing(r.new_ep),
                    count_or_missing(r.ret_rel_ep),
                ),
            )
        else
            (
                add_counts(
                    count_or_missing(r.pulm_labconf_new),
                    count_or_missing(r.pulm_labconf_rec),
                ),
                add_counts(
                    count_or_missing(r.pulm_clindx_new),
                    count_or_missing(r.pulm_clindx_rec),
                ),
                count_or_missing(r.ep_newinc),
            )
        end

        any(ismissing, values) && continue
        lab, clinical, ep = values
        total = lab + clinical + ep

        push!(out, (y, lab, clinical, ep, total))
    end

    return check_years(out)
end

function site_checks(sites, national)
    out = innerjoin(
        sites,
        rename(
            select(national, :year, :cases),
            :cases => :national_total,
        );
        on=:year,
    )

    out.difference = out.total .- out.national_total
    return select(out, :year, :total,
        :national_total, :difference)
end


function historical_presentation(raw)
    categories = ("new_sp", "new_sn", "new_ep")

    ages = (
        ("0-4", "04"),
        ("5-14", "514"),
        ("15-24", "1524"),
        ("25-34", "2534"),
        ("35-44", "3544"),
        ("45-54", "4554"),
        ("55-64", "5564"),
        ("65+", "65"),
    )

    out = DataFrame(
        year=Int[],
        age_group=String[],
        who_category=String[],
        cases=Int[],
    )

    for r in eachrow(raw)
        y = Int(r.year)
        2006 <= y <= 2012 || continue

        for category in categories
            for (age, suffix) in ages

                # Smear-positive childhood data were
                # reported as a combined group in 2006.
                if y == 2006 &&
                    category == "new_sp" &&
                    age in ("0-4", "5-14")
                    continue
                end

                male = count_or_missing(
                    r[Symbol(category * "_m" * suffix)]
                )
                female = count_or_missing(
                    r[Symbol(category * "_f" * suffix)]
                )

                cases = add_counts(male, female)
                ismissing(cases) && continue

                push!(out, (y, age, category, cases))
            end

            if y == 2006 && category == "new_sp"
                cases = add_counts(
                    count_or_missing(r.new_sp_m014),
                    count_or_missing(r.new_sp_f014),
                )

                if !ismissing(cases)
                    push!(out, (
                        y, "0-14", category, cases
                    ))
                end
            end
        end
    end

    sort!(out, [:year, :who_category, :age_group])
    return out
end

function historical_presentation_checks(data, raw)
    out = DataFrame(
        year=Int[],
        who_category=String[],
        age_total=Int[],
        reported_total=Union{Missing,Int}[],
        difference=Union{Missing,Int}[],
    )

    for g in groupby(data, [:year, :who_category])
        y = first(g.year)
        category = first(g.who_category)

        r = only(filter(:year => ==(y), raw))
        reported = count_or_missing(
            r[Symbol(category)]
        )

        age_total = sum(g.cases)
        difference = ismissing(reported) ?
                     missing : age_total - reported

        push!(out, (
            y, category, age_total,
            reported, difference
        ))
    end

    sort!(out, [:year, :who_category])
    return out
end

function historical_site_by_age(presentation)
    data = filter(
        :year => y -> 2007 <= y <= 2012,
        presentation,
    )

    wide = unstack(
        data,
        [:year, :age_group],
        :who_category,
        :cases,
    )

    # Require complete historical presentation records.
    nrow(wide) == 48 ||
        error("Expected 48 year-age combinations")

    for c in (:new_sp, :new_sn, :new_ep)
        any(ismissing, wide[!, c]) &&
            error("Missing historical category: $c")
    end

    wide.pulmonary = wide.new_sp .+ wide.new_sn
    wide.extrapulmonary = wide.new_ep
    wide.total = wide.pulmonary .+ wide.extrapulmonary

    ages = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+",
    ]

    wide.age_order = [
        findfirst(==(a), ages) for a in wide.age_group
    ]

    sort!(wide, [:year, :age_order])

    return select(
        wide, :year, :age_group,
        :pulmonary, :extrapulmonary, :total,
    )
end

function historical_pulmonary_fraction(site_data)
    out = combine(
        groupby(site_data, :age_group),
        :pulmonary => sum => :pulmonary,
        :extrapulmonary => sum => :extrapulmonary,
        :total => sum => :total,
    )

    out.pulmonary_fraction =
        out.pulmonary ./ out.total

    ages = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+",
    ]

    out.age_order = [
        findfirst(==(a), ages) for a in out.age_group
    ]

    sort!(out, :age_order)
    select!(out, Not(:age_order))

    return out
end

function build_surveillance(rawdir, outdir)
    notifications = check_years(
        kiribati(joinpath(rawdir, "notifications.csv"))
    )
    outcomes = check_years(
        kiribati(joinpath(rawdir, "outcomes.csv"))
    )
    age_sex = kiribati(
        joinpath(rawdir, "outcomes_age_sex.csv")
    )

    all_outcomes = treatment_outcomes(outcomes)
    historical = historical_presentation(notifications)
    historical_sites = historical_site_by_age(historical)

    tables = (
        tb_notifications_total=
        national_notifications(notifications),
        tb_notifications_by_age=
        age_notifications(notifications),
        tb_recurrent_notifications=
        recurrent_notifications(notifications),
        tb_treatment_outcomes=
        all_outcomes,
        tb_treatment_outcomes_main=
        main_treatment_outcomes(all_outcomes),
        tb_treatment_outcomes_age_sex=
        age_sex_outcomes(age_sex),
        tb_notifications_by_site=
        site_notifications(notifications),
        tb_historical_new_notifications_by_age_and_type=
        historical,
        tb_new_notifications_by_site_age_2007_2012=
        historical_sites,
        tb_pulmonary_fraction_by_age_2007_2012=
        historical_pulmonary_fraction(historical_sites),
    )

    checks = (
        notification_checks=notification_checks(
            tables.tb_notifications_total,
            tables.tb_notifications_by_age,
        ),
        treatment_checks=treatment_checks(
            tables.tb_treatment_outcomes_main,
        ),
        site_checks=site_checks(
            tables.tb_notifications_by_site,
            tables.tb_notifications_total,
        ),
        historical_presentation_checks=
        historical_presentation_checks(
            tables.tb_historical_new_notifications_by_age_and_type,
            notifications,
        ),
    )

    mkpath(outdir)
    for (name, table) in pairs(tables)
        CSV.write(joinpath(outdir, "$name.csv"), table)
    end

    checkdir = joinpath(outdir, "validation")
    mkpath(checkdir)
    for (name, table) in pairs(checks)
        CSV.write(joinpath(checkdir, "$name.csv"), table)
    end

    return (; tables..., checks...)
end


function country_burden_estimates(raw)
    fields = [
        :year,
        :e_pop_num,
        :e_inc_num, :e_inc_num_lo, :e_inc_num_hi,
        :e_inc_100k, :e_inc_100k_lo, :e_inc_100k_hi,
        :e_mort_num, :e_mort_num_lo, :e_mort_num_hi,
        :e_mort_100k, :e_mort_100k_lo, :e_mort_100k_hi,
        :e_mort_exc_tbhiv_num,
        :e_mort_tbhiv_num,
        :cfr, :cfr_lo, :cfr_hi,
        :c_cdr, :c_cdr_lo, :c_cdr_hi,
    ]

    df = select(raw, fields)
    check_years(df)

    for (best, lo, hi) in (
        (:e_inc_num, :e_inc_num_lo, :e_inc_num_hi),
        (:e_inc_100k, :e_inc_100k_lo, :e_inc_100k_hi),
        (:e_mort_num, :e_mort_num_lo, :e_mort_num_hi),
        (:e_mort_100k, :e_mort_100k_lo, :e_mort_100k_hi),
        (:cfr, :cfr_lo, :cfr_hi),
        (:c_cdr, :c_cdr_lo, :c_cdr_hi),
    )
        for r in eachrow(df)
            any(ismissing, (r[best], r[lo], r[hi])) &&
                continue

            0 <= r[lo] <= r[best] <= r[hi] ||
                error("Invalid WHO interval: $best, $(r.year)")
        end
    end

    return df
end

function build_country_burden(rawdir, outdir)
    raw = kiribati(
        joinpath(rawdir, "burden_countries.csv")
    )

    df = country_burden_estimates(raw)

    mkpath(outdir)
    CSV.write(
        joinpath(outdir, "tb_burden_national.csv"),
        df,
    )

    return df
end


function burden_estimates(raw)
    df = select(
        raw,
        :year, :measure, :unit, :age_group,
        :sex, :risk_factor, :best, :lo, :hi,
    )

    # Retain the original WHO estimates and categories.
    all(df.measure .== "inc") ||
        error("Unexpected burden measure")

    all(df.unit .== "num") ||
        error("Unexpected burden unit")

    for r in eachrow(df)
        if !any(ismissing, (r.best, r.lo, r.hi))
            0 <= r.lo <= r.best <= r.hi ||
                error("Invalid incidence interval")
        end
    end

    allunique(df[:, [
        :year, :measure, :unit,
        :age_group, :sex, :risk_factor,
    ]]) || error("Duplicate burden estimates")

    sort!(df, [:year, :age_group, :sex, :risk_factor])
    return df
end

function preventive_treatment_estimates(raw)
    # Preserve original WHO field names and missing values.
    df = select(raw, Not([
        :country, :iso2, :iso3,
        :iso_numeric, :g_whoregion,
    ]))

    return check_years(df)
end

function build_estimates(rawdir, outdir)
    burden = burden_estimates(
        kiribati(joinpath(rawdir, "burden_age_sex.csv"))
    )

    preventive = preventive_treatment_estimates(
        kiribati(joinpath(rawdir, "preventive_treatment.csv"))
    )

    mkpath(outdir)

    CSV.write(
        joinpath(outdir, "tb_incidence_age_sex.csv"),
        burden,
    )

    CSV.write(
        joinpath(outdir, "tb_preventive_treatment.csv"),
        preventive,
    )

    return (
        incidence_age_sex=burden,
        preventive_treatment=preventive,
    )
end

end # module
