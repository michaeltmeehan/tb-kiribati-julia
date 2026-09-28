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

# Relative age-specific detection multipliers. These specify only the shape of
# age-specific detection. For each year they are normalised against the model-
# predicted age distribution so that the incidence-weighted overall detection
# fraction exactly matches the WHO-derived value for that year.

const INFANT_TO_ADULT_DETECTION_RATIO =
    0.75

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

# Notification data are available from 2013 onwards. WHO incidence data are
# required for the same years because they provide the fixed overall detection
# fraction used by the notification observation model.

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


function overall_detection_fraction(year)

    rows = INCIDENCE[
        INCIDENCE.year .== year,
        :,
    ]

    isempty(rows) &&
        error("No WHO incidence data for year $year")

    row = only(eachrow(rows))

    incidence_rate = Float64(row.incidence_rate)
    notification_rate = Float64(row.notification_rate)

    if !isfinite(incidence_rate) || incidence_rate <= 0 ||
       !isfinite(notification_rate) || notification_rate <= 0
        error("Invalid WHO incidence/notification rate for year $year")
    end

    q = notification_rate / incidence_rate

    if !isfinite(q) || q <= 0 || q > 1
        error("Invalid overall detection fraction for year $year: $q")
    end

    return q
end


"""
    build_notification_observation(year)

Construct the notification observation used by the negative-binomial count
likelihood for one year.

The returned `design` matrix performs only age-category aggregation. Detection
is applied separately before this matrix is used.

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
            1.0

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
            1.0

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
            1.0

        row[2] =
            1.0

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
                1.0

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
        overall_detection = overall_detection_fraction(year),
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


# WHO incidence estimates are not separate likelihood observations. Their
# central estimates are used only through `overall_detection_fraction(year)`.
