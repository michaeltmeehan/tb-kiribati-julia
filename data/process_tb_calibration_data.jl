using CSV
using DataFrames
using Statistics

const RAW_NOTIFICATION_FILE =
    joinpath(@__DIR__, "who_tb_notifications.csv")

const RAW_INCIDENCE_FILE =
    joinpath(@__DIR__, "GTB_report_2025_incidence.csv")

const DATA_DIR =
    normpath(joinpath(@__DIR__, "..", "data"))

const OUT_NOTIFICATION_FILE =
    joinpath(DATA_DIR, "tb_notifications_by_age.csv")

const OUT_INCIDENCE_FILE =
    joinpath(DATA_DIR, "tb_incidence.csv")


# ---------------------------------------------------------------------------
# Notification data
# ---------------------------------------------------------------------------

notifications_raw = CSV.read(
    RAW_NOTIFICATION_FILE,
    DataFrame,
)

# Keep Kiribati only, just in case the raw file is ever replaced by a
# multi-country WHO export.
notifications_raw = filter(
    :iso3 => ==("KIR"),
    notifications_raw,
)


# Build a long-format dataset.
#
# For children, retain:
#   0-4
#   5-14
#   0-14
#
# The 0-14 aggregate is retained because it remains available in years where
# the finer 0-4 and 5-14 breakdown is missing (e.g. 2024).
#
# Male and female notifications are summed because the transmission model is
# currently not sex-stratified.

notification_specs = [
    ("0-4",   :newrel_m04,   :newrel_f04),
    ("5-14",  :newrel_m514,  :newrel_f514),
    ("0-14",  :newrel_m014,  :newrel_f014),
    ("15-24", :newrel_m1524, :newrel_f1524),
    ("25-34", :newrel_m2534, :newrel_f2534),
    ("35-44", :newrel_m3544, :newrel_f3544),
    ("45-54", :newrel_m4554, :newrel_f4554),
    ("55-64", :newrel_m5564, :newrel_f5564),
    ("65+",   :newrel_m65,   :newrel_f65),
]

notifications = DataFrame(
    year = Int[],
    age_group = String[],
    cases = Union{Missing, Int}[],
)

for row in eachrow(notifications_raw)

    for (age_group, male_col, female_col) in notification_specs

        male = row[male_col]
        female = row[female_col]

        cases =
            if ismissing(male) || ismissing(female)
                missing
            else
                Int(round(male + female))
            end

        push!(
            notifications,
            (
                year = Int(row.year),
                age_group = age_group,
                cases = cases,
            ),
        )
    end
end


# ---------------------------------------------------------------------------
# Internal consistency check for childhood data
# ---------------------------------------------------------------------------

for year in unique(notifications.year)

    dat = filter(:year => ==(year), notifications)

    get_cases(group) = begin
        x = dat[dat.age_group .== group, :cases]
        isempty(x) ? missing : only(x)
    end

    cases_04 = get_cases("0-4")
    cases_514 = get_cases("5-14")
    cases_014 = get_cases("0-14")

    if !ismissing(cases_04) &&
       !ismissing(cases_514) &&
       !ismissing(cases_014)

        if cases_04 + cases_514 != cases_014
            @warn(
                "Childhood notification totals are inconsistent",
                year = year,
                cases_04 = cases_04,
                cases_514 = cases_514,
                cases_014 = cases_014,
            )
        end
    end
end


# ---------------------------------------------------------------------------
# Incidence data
# ---------------------------------------------------------------------------

incidence_raw = CSV.read(
    RAW_INCIDENCE_FILE,
    DataFrame,
)

incidence = DataFrame(
    year = Int.(incidence_raw[!, "Category"]),

    incidence_rate = Float64.(
        incidence_raw[
            !,
            "Estimated TB incidence per 100 000 population",
        ],
    ),

    incidence_lower = Float64.(
        incidence_raw[
            !,
            "Uncertainty interval (low)",
        ],
    ),

    incidence_upper = Float64.(
        incidence_raw[
            !,
            "Uncertainty interval (high)",
        ],
    ),

    notification_rate = Float64.(
        incidence_raw[
            !,
            "New and relapse TB cases notified per 100 000 population",
        ],
    ),
)


# ---------------------------------------------------------------------------
# Derived overall notification / incidence ratio
# ---------------------------------------------------------------------------

incidence.detection_fraction =
    incidence.notification_rate ./ incidence.incidence_rate


# ---------------------------------------------------------------------------
# Save processed files
# ---------------------------------------------------------------------------

mkpath(DATA_DIR)

CSV.write(
    OUT_NOTIFICATION_FILE,
    notifications,
)

CSV.write(
    OUT_INCIDENCE_FILE,
    incidence,
)


println()
println("Saved:")
println("  ", OUT_NOTIFICATION_FILE)
println("  ", OUT_INCIDENCE_FILE)

println()
println("Notification years:")
println(
    minimum(notifications.year),
    "–",
    maximum(notifications.year),
)

println()
println("Incidence years:")
println(
    minimum(incidence.year),
    "–",
    maximum(incidence.year),
)

println()
println(
    "Mean overall notification/incidence ratio = ",
    round(mean(incidence.detection_fraction), digits = 3),
)