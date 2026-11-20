
using Test
using TBKiribatiJulia
using Distributions: Normal, truncated, median

include(joinpath(
    @__DIR__, "..", "..",
    "scripts", "common", "kiribati_setup.jl",
))

@testset "DoddExtended calibration integration" begin

    # Shared demographic and surveillance inputs
    setup = setup_dodd_analysis(
        start_year=1970,
        end_year=2026,
        output_years=2013:2025,
    )

    presentation = load_kiribati_presentation(
        setup.surveillance_dir;
        years=2007:2012,
    ).presentation

    model = DoddExtendedTBModel(
        setup.contact,
        presentation;
        reinfection=true,
    )

    national = filter(
        r -> 1980 <= r.year <= 2025,
        setup.surveillance.national_notifications,
    )

    notifications = combine_notification_observations(
        setup.age_observations,
        national;
        n_age_groups=length(setup.age_labels),
        start_year=1970,
        end_year=2026,
    )

    recurrence = RecurrentTBObservation(
        filter(
            r -> 1996 <= r.year <= 2024,
            setup.surveillance.recurrent_notifications,
        ),
        national,
    )

    deaths = TreatmentDeathObservation(
        filter(
            r -> 2012 <= r.year <= 2023,
            setup.surveillance.treatment_outcomes,
        ),
    )

    observation_model = AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=20,
    )

    problem = CalibrationProblem(
        model,
        setup.demography,
        notifications;
        start_year=1970,
        end_year=2026,
        age_breaks=setup.age_breaks,
        observation_model,
        treatment_deaths=deaths,
        recurrence,
    )

    # Parameter construction
    theta = merge(
        map(median, problem.prior),
        extended_fixed_parameters(),
    )

    params = make_parameters(
        model,
        model_draw(problem, theta),
    )

    @test state_layout(model).nages == 96
    @test all(isfinite, params.presentation)

    # One simulation, all required outputs
    output = predict(problem, theta)

    @test :notifications in keys(output)
    @test :recurrent_notifications in keys(output)

    @test all(isfinite, output.notifications)
    @test all(isfinite, output.recurrent_notifications)

    @test all(output.notifications .>= -1e-8)
    @test all(output.recurrent_notifications .>= -1e-8)

    # Recurrence is a subset of notifications
    @test all(
        output.recurrent_notifications .<=
            output.notifications .+ 1e-5
    )

    # Likelihood components
    ll_notifications = notification_loglikelihood(
        problem.observation_model,
        problem.observations,
        output,
        problem.age_breaks,
        theta,
    )

    ll_deaths = treatment_death_loglikelihood(
        problem.treatment_deaths,
        theta,
    )

    ll_recurrence = recurrence_loglikelihood(
        problem.recurrence,
        output,
    )

    @test isfinite(ll_notifications)
    @test isfinite(ll_deaths)
    @test isfinite(ll_recurrence)

    # Combined likelihood must equal component sum
    ll_total = calibration_loglikelihood(
        problem,
        theta;
        output=output,
    )

    @test ll_total ≈
        ll_notifications + ll_deaths + ll_recurrence

    @test isfinite(calibration_logprior(
        problem, theta
    ))

    @test isfinite(calibration_logposterior(
        problem, theta
    ))
end
