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

    # Age-specific population for this year
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

        rate =
            1e5 * cases / population_grouped[g]

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
# Average across years
# ---------------------------------------------------------------------------

mean_rates = combine(
    groupby(rates, :age_group),
    :notification_rate => mean => :mean_rate,
    :notification_rate => std => :sd_rate,
)

# Preserve age order
mean_rates.order = [
    findfirst(==(x), AGE_LABELS)
    for x in mean_rates.age_group
]

sort!(mean_rates, :order)


println(mean_rates)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = Figure(size = (900, 550))

ax = Axis(
    fig[1, 1],
    xlabel = "Age group",
    ylabel = "TB notifications per 100,000",
    title = "Mean age-specific TB notification rate, Kiribati 2013–2023",
)

barplot!(
    ax,
    1:nrow(mean_rates),
    mean_rates.mean_rate,
)

ax.xticks = (
    1:nrow(mean_rates),
    mean_rates.age_group,
)

fig