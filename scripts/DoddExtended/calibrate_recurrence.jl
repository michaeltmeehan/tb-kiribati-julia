
using TBKiribatiJulia
using CSV, DataFrames, Random
using Distributions: Normal, truncated, median

include(joinpath(
    @__DIR__, "..", "common", "kiribati_setup.jl"
))

const START_YEAR = 1970
const END_YEAR = 2026

data_dir = normpath(joinpath(@__DIR__, "..", "..", "data"))
file(name) = joinpath(data_dir, name)

age_breaks = [0, 5, 15, 25, 35, 45, 55, 65, 96]
age_labels = [
    "0-4", "5-14", "15-24", "25-34",
    "35-44", "45-54", "55-64", "65+"
]

# ------------------------------------------------------------
# Build model and load observations
# ------------------------------------------------------------

age_file = file("tb_notifications_by_age.csv")

age_years = sort(unique(
    Int(r.year)
    for r in CSV.File(age_file)
    if 2013 <= r.year < END_YEAR
))

setup = setup_dodd_analysis(
    start_year=START_YEAR,
    end_year=END_YEAR,
    output_years=age_years,
)

presentation_data = load_kiribati_presentation(
    data_dir;
    years=2007:2012,
)

model = DoddExtendedTBModel(
    setup.model.contact,
    presentation_data.presentation;
    reinfection=true,
)

national_counts = load_annual_counts(
    file("tb_notifications_total.csv");
    years=1980:2025,
)

age_observations = load_notification_observations(
    age_file;
    age_labels=age_labels,
    years=age_years,
)

notifications = combine_notification_observations(
    age_observations,
    national_counts;
    n_age_groups=8,
    start_year=START_YEAR,
    end_year=END_YEAR,
)

recurrence = RecurrentTBObservation(
    load_annual_counts(
        file("tb_recurrent_notifications.csv");
        years=1996:2024,
    ),
    national_counts,
)

treatment_deaths = TreatmentDeathObservation(
    load_treatment_outcomes(
        file("tb_treatment_outcomes_main.csv");
        years=2012:2023,
    ),
)

observation_model = AnnualRandomEffectNB2Observation(
    truncated(Normal(0.0, 0.3), 0.0, Inf),
    truncated(Normal(0.0, 0.3), 0.0, Inf);
    quadrature_order=20,
)

function make_problem(recurrence_observation)
    return CalibrationProblem(
        model,
        setup.demography,
        notifications;
        start_year=START_YEAR,
        end_year=END_YEAR,
        age_breaks=age_breaks,
        observation_model=observation_model,
        treatment_deaths=treatment_deaths,
        recurrence=recurrence_observation,
    )
end

problem_total = make_problem(nothing)
problem_joint = make_problem(recurrence)

# ------------------------------------------------------------
# Previous Dodd MAP estimates
# ------------------------------------------------------------

old_map = (
    beta=14.989057115746554,
    psi=0.1915448764038891,
    Lambda0=0.013594088697539785,
    kappa=1.8375992880033918,
    epsilon=0.05829461327951087,
    epsilon_child=0.18998632683550082,
    nu=0.004735410027833472,
    omega=0.023197770195072527,
    detection_logitK=0.17097371093096153,
    detection_trend=0.015264041904355251,
    detection_or_0_4=0.5741766113957257,
    detection_or_5_14=0.8184140987209807,
    duration=2.838468768280723,
    untreated_cfr=0.4281350703008834,
    treatment_cfr=0.06089347940253978,
    sigma_obs=0.2882986429088734,
    tau_year=0.1700779489630453,
)

theta_map = merge(
    map(median, problem_joint.prior),
    old_map,
    extended_fixed_parameters(),
)

# ------------------------------------------------------------
# Recurrence diagnostics
# ------------------------------------------------------------

function recurrence_table(observation, output)
    rows = NamedTuple[]

    for r in observation.rows
        j = findfirst(==(r.year), output.years)

        isnothing(j) && error("Missing year $(r.year)")

        predicted_total = sum(output.notifications[:, j])
        predicted_rec = sum(
            output.recurrent_notifications[:, j]
        )

        push!(rows, (
            year=r.year,
            observed_total=r.total,
            observed_recurrent=r.recurrent,
            observed_fraction=r.recurrent / r.total,
            predicted_total=predicted_total,
            predicted_recurrent=predicted_rec,
            predicted_fraction=predicted_rec / predicted_total,
        ))
    end

    return DataFrame(rows)
end

function print_recurrence_diagnostics(label, theta)
    output = predict(problem_joint, theta)
    table = recurrence_table(recurrence, output)

    println("\n", label)

    show(
        stdout,
        MIME"text/plain"(),
        filter(
            r -> r.year in (1996, 2000, 2013, 2020, 2024),
            table,
        );
        allrows=true,
    )
    println()

    println(
        "Recurrence log likelihood: ",
        recurrence_loglikelihood(recurrence, output),
    )

    return (output=output, table=table)
end

previous_map_diagnostics =
    print_recurrence_diagnostics(
        "Previous Dodd MAP, extended model",
        theta_map,
    )

# ------------------------------------------------------------
# Calibration: fix poorly identified new parameters
# ------------------------------------------------------------

fixed = merge(
    extended_fixed_parameters(),
    (
        duration=3.0,
        treatment_cfr=304 / 4928,
    ),
)

initial = merge(theta_map, fixed)

println("\nFitting total notifications...")

fit_total = fit_map(
    problem_total;
    fixed=fixed,
    initial=initial,
    nstarts=1,
    iterations=3000,
    rng=MersenneTwister(42),
)

println(
    "Total-only MAP likelihood: ",
    fit_total.log_likelihood,
)

total_fit_diagnostics =
    print_recurrence_diagnostics(
        "Extended MAP without recurrence in likelihood",
        fit_total.parameters,
    )

println("\nFitting total + recurrent notifications...")

fit_joint = fit_map(
    problem_joint;
    fixed=fixed,
    initial=fit_total.parameters,
    nstarts=1,
    iterations=3000,
    rng=MersenneTwister(43),
)

println(
    "Joint MAP likelihood: ",
    fit_joint.log_likelihood,
)

joint_fit_diagnostics =
    print_recurrence_diagnostics(
        "Extended MAP with recurrence in likelihood",
        fit_joint.parameters,
    )

# ------------------------------------------------------------
# Save diagnostic comparisons
# ------------------------------------------------------------

results_dir = normpath(joinpath(
    @__DIR__, "..", "..",
    "results", "DoddExtended", "recurrence"
))

mkpath(results_dir)

for (name, diagnostic) in (
    ("previous_map", previous_map_diagnostics),
    ("total_only", total_fit_diagnostics),
    ("joint", joint_fit_diagnostics),
)
    CSV.write(
        joinpath(results_dir, name * "_recurrence.csv"),
        diagnostic.table,
    )
end

DataFrame(
    parameter=String.(collect(keys(fit_joint.parameters))),
    total_only=Float64.(collect(values(fit_total.parameters))),
    joint=Float64.(collect(values(fit_joint.parameters))),
) |> df -> CSV.write(
    joinpath(results_dir, "parameter_comparison.csv"),
    df,
)

println("\nConvergence:")
println("Total-only: ", fit_total.converged)
println("Joint:      ", fit_joint.converged)
