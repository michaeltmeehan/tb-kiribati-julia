# DoddExtended six-parameter MAP optimisation stability.
# Place at scripts/DoddExtended/check_map_stability.jl
# Run from repository root:
#   include("scripts/DoddExtended/check_map_stability.jl")
# Independent, explicitly defined starts. No change to epidemiological initialisation.

using TBKiribatiJulia
using CSV, DataFrames, Random
using Distributions: Normal, median, truncated
using Optim

let
    TB = TBKiribatiJulia
    rng = MersenneTwister(20261125)
    iterations = parse(Int, get(ENV, "TB_STABILITY_ITERATIONS", "1000"))
    result_dir = normpath(joinpath(@__DIR__, "..", "..", "results",
        "DoddExtended", "map_stability"))
    mkpath(result_dir)

    include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))
    setup = setup_dodd_analysis(
        start_year=1970, end_year=2026, output_years=2013:2024)
    presentation = TB.load_kiribati_presentation(
        setup.surveillance_dir; years=2007:2012)
    model = TB.DoddExtendedTBModel(
        setup.contact, presentation.presentation; reinfection=true)

    national = filter(r -> 1980 <= r.year <= 2025,
        setup.surveillance.national_notifications)
    notifications = TB.combine_notification_observations(
        setup.age_observations, national;
        n_age_groups=length(setup.age_labels),
        start_year=1970, end_year=2026)
    recurrence = TB.RecurrentTBObservation(
        filter(r -> 1996 <= r.year <= 2024,
            setup.surveillance.recurrent_notifications), national)
    site = TB.SiteTBObservation(
        filter(r -> 1970 <= r.year < 2026,
            setup.surveillance.site_notifications))
    treatment_deaths = TB.TreatmentDeathObservation(
        filter(r -> 2012 <= r.year <= 2023,
            setup.surveillance.treatment_outcomes))

    obs_model = TB.AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=40)
    problem = TB.CalibrationProblem(
        model, setup.demography, notifications;
        start_year=1970, end_year=2026,
        age_breaks=setup.age_breaks,
        observation_model=obs_model,
        recurrence=recurrence, site=site,
        treatment_deaths=treatment_deaths,
        solver=TB.Tsit5())

    free_names = (:beta, :Lambda0, :nu, :omega,
        :detection_logitK, :r_SP)
    @assert all(k -> haskey(problem.prior, k), free_names)

    defaults = merge(map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (detection_trend=0.0,
            sigma_obs=0.30, tau_year=0.18))
    fixed_names = Tuple(k for k in keys(problem.prior) if !(k in free_names))
    fixed = NamedTuple{fixed_names}(
        Tuple(getproperty(defaults, k) for k in fixed_names))

    reference = (
        beta=10.985230913865225,
        Lambda0=0.006339206566999288,
        nu=0.015186477942370179,
        omega=0.0018303349840744073,
        detection_logitK=-0.002478460182136632,
        r_SP=3.488980445181244,
    )

    # Local starting points are perturbed on the natural (log) scale for
    # positive parameters, and additively for the detection logit.
    positive_names = (:beta, :Lambda0, :nu, :omega, :r_SP)
    function local_start(ref, random_state; log_sd=0.35, logit_sd=0.4)
        vals = (
            (k == :detection_logitK ?
             getproperty(ref, k) + logit_sd * randn(random_state) :
             getproperty(ref, k) * exp(log_sd * randn(random_state)))
            for k in free_names
        )
        return NamedTuple{free_names}(Tuple(vals))
    end

    # A genuinely dispersed start uses the independent joint prior.
    function prior_start(prior, random_state)
        return NamedTuple{free_names}(
            Tuple(rand(random_state, getproperty(prior, k)) for k in free_names))
    end

    starts = [
        ("reference_MAP", reference),
        ("prior_medians", NamedTuple{free_names}(
            Tuple(getproperty(defaults, k) for k in free_names))),
        ("near_MAP_1", local_start(reference, rng)),
        ("near_MAP_2", local_start(reference, rng)),
        ("near_MAP_3", local_start(reference, rng)),
        ("prior_draw_1", prior_start(problem.prior, rng)),
        ("prior_draw_2", prior_start(problem.prior, rng)),
        ("prior_draw_3", prior_start(problem.prior, rng)),
    ]

    # Save starting values before running anything, so interrupted runs
    # still have a complete audit trail.
    start_rows = NamedTuple[]
    for (i, (label, start)) in enumerate(starts)
        for k in free_names
            push!(start_rows, (run=i, label=label, parameter=String(k),
                value=Float64(getproperty(start, k))))
        end
    end
    CSV.write(joinpath(result_dir, "starting_parameters.csv"),
        DataFrame(start_rows))

    summary_rows = NamedTuple[]
    parameter_rows = NamedTuple[]

    println("DoddExtended MAP STABILITY: 8 explicit starts; $iterations iterations each")
    println("Free: ", join(string.(free_names), ", "))
    println("Fixed: all other prior medians; detection trend=0; sigma=0.30; tau=0.18")
    println("Likelihood unchanged; epidemiological initial conditions unchanged")
    println("Output: $result_dir")
    flush(stdout)

    for (i, (label, start)) in enumerate(starts)
        initial = merge(defaults, start)
        println("\n[$i/8] $label")
        println("  Starting: ", join(("$k=$(round(getproperty(start,k),sigdigits=5))" for k in free_names), ", "))
        flush(stdout)

        try
            fit = TB.fit_map(
                problem;
                fixed=fixed, initial=initial,
                optimizer=:nelder_mead,
                iterations=iterations,
                nstarts=1,
                rng=MersenneTwister(20261125 + i))

            theta = fit.parameters
            # Use the same prediction API as the original MAP script.
            out = TB.predict(problem, theta;
                variables=(:population, :incidence, :notifications,
                    :mortality, :latent_prevalence,
                    :recurrent_notifications,
                    :pulmonary_notifications))
            j = findfirst(==(2023), out.years)
            isnothing(j) && error("Predictions did not contain 2023")
            total(k) = sum(@view getproperty(out, k)[:, j])
            pop = total(:population)
            inc = total(:incidence)
            notif = total(:notifications)
            deaths = total(:mortality)

            push!(summary_rows, (
                run=i, label=label, status="ok",
                converged=fit.converged,
                iterations=Optim.iterations(fit.optim_result),
                evaluations=Optim.f_calls(fit.optim_result),
                log_likelihood=fit.log_likelihood,
                log_prior=fit.log_prior,
                log_posterior=fit.log_posterior,
                beta=theta.beta, Lambda0=theta.Lambda0,
                nu=theta.nu, omega=theta.omega,
                detection_logitK=theta.detection_logitK,
                r_SP=theta.r_SP,
                incidence_per_100k=1e5*inc/pop,
                mortality_per_100k=1e5*deaths/pop,
                notifications_2023=notif,
                latent_pct=100*total(:latent_prevalence)/pop,
                recurrent_fraction=notif>0 ? total(:recurrent_notifications)/notif : NaN,
                pulmonary_fraction=notif>0 ? total(:pulmonary_notifications)/notif : NaN,
                error="",
            ))
            for k in free_names
                push!(parameter_rows, (
                    run=i, label=label, parameter=String(k),
                    start=Float64(getproperty(start, k)),
                    fitted=Float64(getproperty(theta, k))))
            end
            println("  Converged=$(fit.converged), iterations=$(Optim.iterations(fit.optim_result)), log posterior=$(round(fit.log_posterior; digits=4))")
            println("  beta=$(round(theta.beta;digits=4)), nu=$(round(theta.nu;digits=6)), omega=$(round(theta.omega;digits=6)), r_SP=$(round(theta.r_SP;digits=4))")
            theta.beta > 40 && println("  WARNING: high-transmission solution (beta > 40)")
        catch err
            println("  ERROR: ", sprint(showerror, err))
            push!(summary_rows, (
                run=i, label=label, status="error",
                converged=false, iterations=missing, evaluations=missing,
                log_likelihood=missing, log_prior=missing,
                log_posterior=missing,
                beta=missing, Lambda0=missing, nu=missing, omega=missing,
                detection_logitK=missing, r_SP=missing,
                incidence_per_100k=missing,
                mortality_per_100k=missing,
                notifications_2023=missing, latent_pct=missing,
                recurrent_fraction=missing, pulmonary_fraction=missing,
                error=sprint(showerror, err),
            ))
        end
        # Checkpoint after every run, including on failure.
        CSV.write(joinpath(result_dir, "run_summary.csv"), DataFrame(summary_rows))
        if !isempty(parameter_rows)
            CSV.write(joinpath(result_dir, "parameter_comparison.csv"),
                DataFrame(parameter_rows))
        end
        flush(stdout)
    end

    summary = DataFrame(summary_rows)
    completed = filter(:status => ==("ok"), summary)
    if nrow(completed) > 0
        best = completed[argmax(completed.log_posterior), :]
        println("\nBEST RESULT: run $(best.run) ($(best.label)), log posterior=$(best.log_posterior)")
        println("Posterior range among completed runs: ",
            extrema(completed.log_posterior))
        println("\nRUN SUMMARY")
        show(stdout, MIME("text/plain"),
            select(summary, :run, :label, :status, :converged,
                :iterations, :log_posterior, :beta, :nu,
                :omega, :r_SP, :incidence_per_100k,
                :mortality_per_100k, :notifications_2023);
            allrows=true, allcols=true)
        println()
    end
    println("Saved: $result_dir")
end
