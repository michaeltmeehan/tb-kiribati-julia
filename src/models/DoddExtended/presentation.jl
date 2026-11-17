
const EXTENDED_AGE_BREAKS = [
    0, 5, 15, 25, 35, 45, 55, 65, 96
]

const EXTENDED_AGE_LABELS = [
    "0-4", "5-14", "15-24", "25-34",
    "35-44", "45-54", "55-64", "65+"
]


"""
Expand observed presentation counts into age-specific
onset probabilities.

Columns are:
    new_sp, new_sn, new_ep
"""
function extended_presentation(
    counts::AbstractMatrix;
    age_breaks=EXTENDED_AGE_BREAKS,
    nages::Integer=96,
)
    breaks = Int.(age_breaks)
    C = Matrix{Float64}(counts)

    size(C) == (length(breaks) - 1, 3) ||
        throw(DimensionMismatch("Invalid count dimensions"))

    first(breaks) == 0 && last(breaks) == nages ||
        throw(ArgumentError("Invalid age breaks"))

    all(diff(breaks) .> 0) ||
        throw(ArgumentError("Age breaks must increase"))

    all(x -> isfinite(x) && x >= 0, C) ||
        throw(ArgumentError("Invalid presentation counts"))

    totals = vec(sum(C; dims=2))

    all(totals .> 0) ||
        throw(ArgumentError("Empty presentation group"))

    fractions = C ./ totals
    presentation = zeros(Float64, nages, 3)

    for a in 1:nages
        age = a - 1
        group = searchsortedlast(breaks, age)

        @views presentation[a, :] .=
            fractions[group, :]
    end

    return presentation
end


"""
Load Kiribati age-specific presentation observations.

Uses the existing cleaned historical WHO dataset,
without imputing missing categories.
"""
function load_kiribati_presentation(
    data_dir::AbstractString;
    years=2007:2012,
    nages::Integer=96,
)
    path = joinpath(
        data_dir,
        "tb_historical_new_notifications_by_age_and_type.csv",
    )

    rows = load_historical_notification_types(
        path; years=years
    )

    categories = Dict(
        "new_sp" => 1,
        "new_sn" => 2,
        "new_ep" => 3,
    )

    counts = zeros(Int, length(EXTENDED_AGE_LABELS), 3)

    for row in rows
        haskey(categories, row.who_category) || continue

        group = findfirst(
            ==(row.age_group),
            EXTENDED_AGE_LABELS,
        )

        isnothing(group) &&
            throw(ArgumentError(
                "Unexpected age group $(row.age_group)"
            ))

        k = categories[row.who_category]
        counts[group, k] += row.cases
    end

    presentation = extended_presentation(
        counts;
        nages=nages,
    )

    return (
        presentation=presentation,
        counts=counts,
        age_breaks=EXTENDED_AGE_BREAKS,
        age_labels=EXTENDED_AGE_LABELS,
    )
end
