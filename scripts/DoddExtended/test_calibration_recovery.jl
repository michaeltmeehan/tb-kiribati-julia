
using TBKiribatiJulia
using Distributions: median
using Random
using DataFrames
using Statistics: mean

const TB = TBKiribatiJulia

include(joinpath(
    @__DIR__, "..", "common", "kiribati_setup.jl"
))

# --------------------------------------------------
# Construct the model
# --------------------------------------------------

setup = setup_dodd_analysis(
    start_year=1970,
    end_year=2026,
    output_years=2013:2024,
)

presentation = TB.load_kiribati_presentation(
    setup.surveillance_dir;
    years=2007:2012,
)

model = TB.DoddExtendedTBModel(
    setup.contact,
    presentation.presentation;
    reinfection=true,
)

# Use the same notification observation structure
# as the real-data calibration.
national = filter(
    r -> 1980 <= r.year <= 2025,
    setup.surveillance.national_notifications,
)

observations = TB.combine_notification_observations(
    setup.age_observations,
    national;
    n_age_groups=length(setup.age_labels),
    start_year=1970,
    end_year=2026,
)

# Independent NB2 initially removes the annual
# random effect as a possible complication.
using Distributions: Normal, truncated

obs_model = TB.NB2Observation(
    truncated(Normal(0, 0.3), 0.0, Inf)
)

base_problem = TB.CalibrationProblem(
    model,
    setup.demography,
    observations;
    start_year=1970,
    end_year=2026,
    age_breaks=setup.age_breaks,
    observation_model=obs_model,
)

# --------------------------------------------------
# Known generating parameters
# --------------------------------------------------

theta_true = map(median, base_problem.prior)

theta_true = merge(theta_true, (
    beta=30.0,
    Lambda0=0.03,
    detection_logitK=-0.1,
    detection_trend=0.015,
    sigma_obs=0.30,
))

free_names = (
    :beta,
    :Lambda0,
    :detection_logitK,
    :detection_trend,
)

# --------------------------------------------------
# Generate synthetic observations
# --------------------------------------------------

pred_true = TB.predict(base_problem, theta_true)

synthetic_obs = map(observations) do obs
    j = findfirst(==(obs.year), pred_true.years)

    counts = Int[
        round(Int, sum(
            pred_true.notifications[g, j]
            for g in group
        ))
        for group in obs.groups
    ]

    merge(obs, (counts=counts,))
end

synthetic_problem = TB.CalibrationProblem(
    model,
    setup.demography,
    synthetic_obs;
    start_year=1970,
    end_year=2026,
    age_breaks=setup.age_breaks,
    observation_model=obs_model,
)

# --------------------------------------------------
# Test 1: observation extraction
# --------------------------------------------------

rows_true = TB.notification_fit_rows(
    synthetic_problem,
    pred_true,
)

errors = [
    abs(r.observed - r.expected)
    for r in rows_true
]

println("\nSYNTHETIC DATA EXTRACTION")
println("Number of observations: ", length(rows_true))
println("Maximum rounding error: ", maximum(errors))

@assert maximum(errors) <= 0.5 + 1e-8

# --------------------------------------------------
# Test 2: likelihood evaluation
# --------------------------------------------------

ll_true = TB.calibration_loglikelihood(
    synthetic_problem,
    theta_true;
    output=pred_true,
)

# Deliberately incorrect detection parameters.
theta_wrong = merge(theta_true, (
    detection_logitK=theta_true.detection_logitK - 2,
))

ll_wrong = TB.calibration_loglikelihood(
    synthetic_problem,
    theta_wrong,
)

println("\nLIKELIHOOD CHECK")
println("Generating parameters: ", ll_true)
println("Incorrect detection:  ", ll_wrong)
println("Difference:           ", ll_true - ll_wrong)

@assert isfinite(ll_true)
@assert ll_true > ll_wrong

# --------------------------------------------------
# Test 3: four-parameter recovery
# --------------------------------------------------

fixed_names = Tuple(
    k for k in keys(synthetic_problem.prior)
          if !(k in free_names)
)

fixed = NamedTuple{fixed_names}(
    Tuple(theta_true[k] for k in fixed_names)
)

recovered = TB.fit_mle(
    synthetic_problem;
    fixed=fixed,
    initial=theta_true,
    optimizer=:nelder_mead,
    iterations=1500,
    nstarts=3,
    rng=MersenneTwister(42),
)

println("\nPARAMETER RECOVERY")

results = DataFrame(
    parameter=String.(free_names),
    generating=[
        theta_true[k] for k in free_names
    ],
    recovered=[
        recovered.parameters[k] for k in free_names
    ],
)

display(results)

println("Converged: ", recovered.converged)
println("Generating log likelihood: ", ll_true)
println("Recovered log likelihood: ",
    recovered.log_likelihood)

# --------------------------------------------------
# Test 4: prediction recovery
# --------------------------------------------------

pred_recovered = TB.predict(
    synthetic_problem,
    recovered.parameters,
)

rows_recovered = TB.notification_fit_rows(
    synthetic_problem,
    pred_recovered,
)

comparison = DataFrame(
    year=[r.year for r in rows_true],
    age_group=[r.age_group for r in rows_true],
    generated=[r.expected for r in rows_true],
    recovered=[r.expected for r in rows_recovered],
)

comparison.absolute_error = abs.(
    comparison.recovered .-
        comparison.generated
)

comparison.relative_error =
    comparison.absolute_error ./
        max.(comparison.generated, 1.0)

println("\nPREDICTION RECOVERY")
println("Mean absolute error: ",
    mean(comparison.absolute_error))
println("Mean relative error: ",
    mean(comparison.relative_error))
println("Maximum absolute error: ",
    maximum(comparison.absolute_error))

display(first(
    sort(comparison, :absolute_error; rev=true),
    10
))
