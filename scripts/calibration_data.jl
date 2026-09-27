using CSV
using DataFrames


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

const DATA_DIR =
    normpath(joinpath(@__DIR__, "..", "data"))

const NOTIFICATION_FILE =
    joinpath(
        DATA_DIR,
        "tb_notifications_by_age.csv",
    )

const INCIDENCE_FILE =
    joinpath(
        DATA_DIR,
        "tb_incidence.csv",
    )


const NOTIFICATIONS = CSV.read(
    NOTIFICATION_FILE,
    DataFrame,
)

const INCIDENCE = CSV.read(
    INCIDENCE_FILE,
    DataFrame,
)


# ---------------------------------------------------------------------------
# Calibration age groups
# ---------------------------------------------------------------------------

const CALIBRATION_AGE_BREAKS = [
    0,
    5,
    15,
    25,
    35,
    45,
    55,
    65,
    96,
]

const CALIBRATION_AGE_LABELS = [
    "0-4",
    "5-14",
    "15-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]


# ---------------------------------------------------------------------------
# Relative detection assumptions
# ---------------------------------------------------------------------------

# Only relative detection rates matter for the notification age-composition
# likelihood.
#
# These are retained separately so that infant and older-child detection
# assumptions can be modified independently if required.

const INFANT_TO_ADULT_DETECTION_RATIO =
    1.0

const CHILD_TO_ADULT_DETECTION_RATIO =
    1.0


const RELATIVE_DETECTION = [
    INFANT_TO_ADULT_DETECTION_RATIO,   # 0-4
    CHILD_TO_ADULT_DETECTION_RATIO,    # 5-14
    1.0,                               # 15-24
    1.0,                               # 25-34
    1.0,                               # 35-44
    1.0,                               # 45-54
    1.0,                               # 55-64
    1.0,                               # 65+
]


# ---------------------------------------------------------------------------
# Calibration years
# ---------------------------------------------------------------------------

# Notification data are available from 2013 onwards. Restrict the incidence
# likelihood to the same period so that both likelihood components describe
# the same calibration era.

const CALIBRATION_YEARS = sort(
    intersect(
        unique(Int.(NOTIFICATIONS.year)),
        unique(Int.(INCIDENCE.year)),
    ),
)


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------

# The calibration model assumes equilibrium demography and constant
# epidemiological parameters. Annual observations are therefore treated as
# repeated observations of a common equilibrium process.

const TINIT =
    1800.0

const TFINAL =
    2024.0


# Parameters that differ from make_parameters defaults but are not currently
# varied during calibration.

const FIXED_PARAMETERS = (
    infectiousness_weights =
        (0.2, 0.5, 0.4, 1.0),
)


# ---------------------------------------------------------------------------
# Notification-data helpers
# ---------------------------------------------------------------------------

function case_count(
    year_data,
    age_group,
)

    values = year_data[
        year_data.age_group .== age_group,
        :cases,
    ]

    isempty(values) &&
        return missing

    return only(values)
end


"""
    build_notification_observation(year)

Construct the notification observation used by the age-composition
likelihood for one year.

The returned `design` matrix maps model-predicted incidence in the eight
calibration age groups onto the notification categories available in that
year.

Separate 0-4 and 5-14 observations are used where both are available.
Otherwise, an observed 0-14 category is represented by a design-matrix row
combining the first two model age groups.
"""
function build_notification_observation(
    year,
)

    year_data = NOTIFICATIONS[
        NOTIFICATIONS.year .== year,
        :,
    ]

    isempty(year_data) &&
        error(
            "No notification data for year $year",
        )

    counts =
        Int[]

    rows =
        Vector{Vector{Float64}}()

    labels =
        String[]

    n_age_groups =
        length(CALIBRATION_AGE_LABELS)


    # -----------------------------------------------------------------------
    # Childhood observations
    # -----------------------------------------------------------------------

    y04 =
        case_count(
            year_data,
            "0-4",
        )

    y514 =
        case_count(
            year_data,
            "5-14",
        )

    y014 =
        case_count(
            year_data,
            "0-14",
        )


    if !ismissing(y04) &&
       !ismissing(y514)

        row =
            zeros(
                Float64,
                n_age_groups,
            )

        row[1] =
            INFANT_TO_ADULT_DETECTION_RATIO

        push!(
            counts,
            Int(y04),
        )

        push!(
            rows,
            row,
        )

        push!(
            labels,
            "0-4",
        )


        row =
            zeros(
                Float64,
                n_age_groups,
            )

        row[2] =
            CHILD_TO_ADULT_DETECTION_RATIO

        push!(
            counts,
            Int(y514),
        )

        push!(
            rows,
            row,
        )

        push!(
            labels,
            "5-14",
        )


    elseif !ismissing(y014)

        row =
            zeros(
                Float64,
                n_age_groups,
            )

        row[1] =
            INFANT_TO_ADULT_DETECTION_RATIO

        row[2] =
            CHILD_TO_ADULT_DETECTION_RATIO

        push!(
            counts,
            Int(y014),
        )

        push!(
            rows,
            row,
        )

        push!(
            labels,
            "0-14",
        )


    else

        error(
            "No usable childhood notification data for year $year",
        )
    end


    # -----------------------------------------------------------------------
    # Adult observations
    # -----------------------------------------------------------------------

    for g in 3:n_age_groups

        age_group =
            CALIBRATION_AGE_LABELS[g]

        y =
            case_count(
                year_data,
                age_group,
            )

        if !ismissing(y)

            row =
                zeros(
                    Float64,
                    n_age_groups,
                )

            row[g] =
                RELATIVE_DETECTION[g]

            push!(
                counts,
                Int(y),
            )

            push!(
                rows,
                row,
            )

            push!(
                labels,
                age_group,
            )
        end
    end


    # Each row corresponds to one observed notification category.
    design =
        reduce(
            vcat,
            permutedims.(rows),
        )


    return (
        year = year,
        counts = counts,
        design = design,
        labels = labels,
    )
end


# ---------------------------------------------------------------------------
# Processed notification observations
# ---------------------------------------------------------------------------

const NOTIFICATION_OBSERVATIONS = [
    build_notification_observation(year)
    for year in CALIBRATION_YEARS
]


# ---------------------------------------------------------------------------
# Processed WHO incidence observations
# ---------------------------------------------------------------------------

const INCIDENCE_OBSERVATIONS = [
    (
        year =
            Int(row.year),

        observed =
            Float64(row.incidence_rate),

        lower =
            Float64(row.incidence_lower),

        upper =
            Float64(row.incidence_upper),
    )

    for row in eachrow(INCIDENCE)

    if Int(row.year) in CALIBRATION_YEARS
]