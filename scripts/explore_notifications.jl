using CSV
using DataFrames
using Statistics
using CairoMakie

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))

notifications = CSV.read(
    joinpath(DATA_DIR, "tb_notifications_by_age.csv"),
    DataFrame,
)

const AGE_BREAKS = [0, 5, 15, 25, 35, 45, 55, 65, 96]

const AGE_LABELS = [
    "0-4",
    "5-14",
    "15-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]

# Exclude 2024 because 0-4 and 5-14 are not separately available
const YEARS = 2013.:2023.


# ---------------------------------------------------------------------------
# Calculate annual notification rates
# ---------------------------------------------------------------------------

rates = DataFrame(
    year = Float64[],
    age_group = String[],
    notification_rate = Float64[],
)

for year in YEARS

    population = get_population(year)

    population_grouped = aggregate_age_groups(
        population,
        AGE_BREAKS,
    )

    year_data = notifications[
        notifications.year .== year,
        :,
    ]

    for (g, age_group) in enumerate(AGE_LABELS)

        values = year_data[
            year_data.age_group .== age_group,
            :cases,
        ]

        isempty(values) && continue

        cases = only(values)

        ismissing(cases) && continue

        rate = 1e5 * cases / population_grouped[g]

        push!(
            rates,
            (
                year = year,
                age_group = age_group,
                notification_rate = rate,
            ),
        )
    end
end


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------

# Convert age groups to plotting positions
age_index = Dict(label => i for (i, label) in enumerate(AGE_LABELS))

x = [age_index[a] for a in rates.age_group]
y = rates.notification_rate

fig = Figure(size = (1000, 600))

ax = Axis(
    fig[1, 1],
    xlabel = "Age group",
    ylabel = "TB notifications per 100,000",
    title = "Age-specific TB notification rates, Kiribati 2013–2023",
    xticks = (1:length(AGE_LABELS), AGE_LABELS),
)

CairoMakie.boxplot!(
    ax,
    x,
    y,
)

# Optional: overlay the annual observations as points
jitter_width = 0.12

x_jittered =
    x .+
    jitter_width .* (2 .* rand(length(x)) .- 1)

CairoMakie.scatter!(
    ax,
    x_jittered,
    y,
    markersize = 8,
)

fig