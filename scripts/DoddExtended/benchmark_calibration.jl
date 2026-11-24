
using TBKiribatiJulia
using Distributions: Normal, truncated, median
using Statistics: median as stat_median

const TB = TBKiribatiJulia

include(joinpath(
    @__DIR__, "..", "common", "kiribati_setup.jl"
))

# --------------------------------------------------
# Reconstruct Experiment A
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

national = filter(
    r -> 1980 <= r.year <= 2025,
    setup.surveillance.national_notifications,
)

notifications = TB.combine_notification_observations(
    setup.age_observations,
    national;
    n_age_groups=length(setup.age_labels),
    start_year=1970,
    end_year=2026,
)

observation_model = TB.AnnualRandomEffectNB2Observation(
    truncated(Normal(0.0, 0.3), 0.0, Inf),
    truncated(Normal(0.0, 0.3), 0.0, Inf);
    quadrature_order=40,
)

problem = TB.CalibrationProblem(
    model,
    setup.demography,
    notifications;
    start_year=1970,
    end_year=2026,
    age_breaks=setup.age_breaks,
    observation_model=observation_model,
)

# --------------------------------------------------
# Parameters from the previous Experiment A fit
# --------------------------------------------------

theta = merge(
    map(median, problem.prior),
    (
        beta=109.39541347262308,
        Lambda0=0.6130527202800097,
        detection_logitK=4.639471845284076,
        detection_trend=-0.015084062050421084,
        omega=0.001,
        sigma_obs=0.30,
        tau_year=0.18,
    ),
)

# --------------------------------------------------
# Benchmark
# --------------------------------------------------

println("\nCompiling / warming up likelihood...")
@time ll = TB.calibration_loglikelihood(problem, theta)

println("Log likelihood: ", ll)

println("\nTiming five additional evaluations...")

times = Float64[]

for i in 1:5
    elapsed = @elapsed begin
        ll = TB.calibration_loglikelihood(problem, theta)
    end

    push!(times, elapsed)

    println(
        "Evaluation $i: ",
        round(elapsed; digits=3),
        " seconds"
    )
end

println("\nBENCHMARK SUMMARY")
println("Median evaluation time: ",
    round(stat_median(times); digits=3), " s")
println("Minimum evaluation time: ",
    round(minimum(times); digits=3), " s")
println("Maximum evaluation time: ",
    round(maximum(times); digits=3), " s")

println("\nEstimated runtime (excluding overhead):")

for n in (100, 500, 1000, 2000)
    minutes = n * stat_median(times) / 60

    println(
        "$n evaluations: ",
        round(minutes; digits=2),
        " minutes"
    )
end
