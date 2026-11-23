
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

    site = SiteTBObservation(filter(
        r -> 2013 <= r.year <= 2025,
        setup.surveillance.site_notifications,
    ))

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
        site,
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
    for P in (
        params.presentation_fast,
        params.presentation_slow,
        params.presentation_relapse,
    )
        @test all(isfinite, P)
    end

    # One simulation, all required outputs
    output = predict(problem, theta)

    @test :pulmonary_notifications in keys(output)
    @test :extrapulmonary_notifications in keys(output)

    @test all(isfinite, output.pulmonary_notifications)
    @test all(isfinite, output.extrapulmonary_notifications)

    @test all(isapprox.(
        output.notifications,
        output.pulmonary_notifications .+
            output.extrapulmonary_notifications;
        atol=1e-5,
    ))

    # Full diagnostic extraction
    full = predict(problem, theta; variables=nothing)

    @test :incidence in keys(full)
    @test :mortality in keys(full)
    @test :reinfections in keys(full)
    @test :active_prevalence in keys(full)
    @test :latent_prevalence in keys(full)

    @test full.notifications ≈ output.notifications

    @test calibration_loglikelihood(
        problem, theta; output=full
    ) ≈ calibration_loglikelihood(
        problem, theta; output=output
    )

    # Selected outputs
    selected = predict(
        problem, theta;
        variables=(:incidence,),
    )

    @test :incidence in keys(selected)
    @test :recurrent_notifications in keys(selected)
    @test :pulmonary_notifications in keys(selected)
    @test :extrapulmonary_notifications in keys(selected)

    @test_throws ArgumentError predict(
        problem, theta; years=[2024]
    )

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

    ll_site = site_loglikelihood(problem.site, output)
    @test isfinite(ll_site)

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
        ll_notifications + ll_deaths + ll_recurrence + ll_site

    @test isfinite(calibration_logprior(
        problem, theta
    ))

    @test isfinite(calibration_logposterior(
        problem, theta
    ))
end
