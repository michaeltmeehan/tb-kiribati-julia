
using TBKiribatiJulia
using DataFrames, CSV

const TB = TBKiribatiJulia

problem = problems[1]
theta_base = fits[1].parameters

omega_grid = [
    0.0,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.010,
    0.015,
    theta_base.omega,
]

variables = (
    :incidence,
    :notifications,
    :recurrent_notifications,
    :relapse,
)

results = NamedTuple[]

for omega in omega_grid
    theta = merge(theta_base, (omega=omega,))

    output = TB.predict(
        problem, theta;
        variables=variables,
    )

    for year in (2013, 2019, 2023, 2024)
        j = findfirst(==(year), output.years)

        incidence = sum(output.incidence[:, j])
        notifications = sum(output.notifications[:, j])
        recurrent = sum(
            output.recurrent_notifications[:, j]
        )

        push!(results, (
            omega=omega,
            year=year,
            incidence=incidence,
            notifications=notifications,
            recurrent_notifications=recurrent,
            recurrent_fraction=recurrent / notifications,
            relapse_onset=sum(output.relapse[:, j]),
        ))
    end
end

df = DataFrame(results)

dir = joinpath(
    "results", "DoddExtended",
    "calibration_component_comparison",
)
mkpath(dir)

CSV.write(joinpath(dir, "relapse_grid.csv"), df)

println("\n2023 RELAPSE SENSITIVITY")
display(filter(r -> r.year == 2023, df))

println("\n2024 RELAPSE SENSITIVITY")
display(filter(r -> r.year == 2024, df))
