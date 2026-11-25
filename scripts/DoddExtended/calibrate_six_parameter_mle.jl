# Six-parameter maximum-likelihood calibration of DoddExtended.
# Place at scripts/DoddExtended/calibrate_six_parameter_mle.jl
# From the repository root (Julia REPL):
#   include("scripts/DoddExtended/calibrate_six_parameter_mle.jl")
# PowerShell optional controls (before starting Julia):
#   $env:TB_MLE_ITERATIONS="1000"
#   $env:TB_MLE_STARTS="1"
#   $env:TB_MLE_SEED="20261125"
# No WHO incidence or mortality estimates enter the likelihood.
# Changes relative to the working MAP script: fit_mle, MAP starting point,
# 1000 default iterations, and separate output directory.

using TBKiribatiJulia
using CSV, DataFrames, Random
using Distributions: Normal, median, truncated
using Optim

let
    TB = TBKiribatiJulia
    start_year, end_year = 1970, 2026
    iterations = parse(Int, get(ENV, "TB_MLE_ITERATIONS", "1000"))
    nstarts = parse(Int, get(ENV, "TB_MLE_STARTS", "1"))
    seed = parse(Int, get(ENV, "TB_MLE_SEED", "20261125"))
    result_dir = normpath(joinpath(@__DIR__, "..", "..", "results",
        "DoddExtended", "six_parameter_mle"))
    mkpath(result_dir)

    include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))
    setup = setup_dodd_analysis(
        start_year=start_year, end_year=end_year, output_years=2013:2024)
    pres = TB.load_kiribati_presentation(
        setup.surveillance_dir; years=2007:2012)
    model = TB.DoddExtendedTBModel(
        setup.contact, pres.presentation; reinfection=true)

    national = filter(r -> 1980 <= r.year <= 2025,
        setup.surveillance.national_notifications)
    notifications = TB.combine_notification_observations(
        setup.age_observations, national;
        n_age_groups=length(setup.age_labels),
        start_year=start_year, end_year=end_year)
    recurrence = TB.RecurrentTBObservation(
        filter(r -> 1996 <= r.year <= 2024,
            setup.surveillance.recurrent_notifications), national)
    site = TB.SiteTBObservation(
        filter(r -> start_year <= r.year < end_year,
            setup.surveillance.site_notifications))
    treatment_deaths = TB.TreatmentDeathObservation(
        filter(r -> 2012 <= r.year <= 2023,
            setup.surveillance.treatment_outcomes))

    obs_model = TB.AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=40)

    # Tsit5 avoids the Rosenbrock automatic-differentiation workspace error.
    problem = TB.CalibrationProblem(
        model, setup.demography, notifications;
        start_year=start_year, end_year=end_year,
        age_breaks=setup.age_breaks, observation_model=obs_model,
        recurrence=recurrence, site=site,
        treatment_deaths=treatment_deaths,
        solver=TB.Tsit5())

    free_names = (:beta, :Lambda0, :nu, :omega,
        :detection_logitK, :r_SP)
    missing = filter(k -> !haskey(problem.prior, k), free_names)
    isempty(missing) || error("Missing priors: $missing. Update parameters.jl")

    # Everything except the six free parameters is fixed at its prior median.
    # The detection trend and observation dispersion are explicit assumptions.
    defaults = merge(map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (detection_trend=0.0,
            sigma_obs=0.30, tau_year=0.18))
    fixed_names = Tuple(k for k in keys(problem.prior) if !(k in free_names))
    fixed = NamedTuple{fixed_names}(
        Tuple(getproperty(defaults, k) for k in fixed_names))
    @assert all(k -> !haskey(fixed, k), free_names)

    # Exact stable MAP fit: use as the MLE starting point.
    # Priors are used ONLY to define the fixed natural-history medians.
    # They are NOT used in the MLE objective.
    initial = merge(defaults, (
        beta=10.985230913865225,
        Lambda0=0.006339206566999288,
        nu=0.015186477942370179,
        omega=0.0018303349840744073,
        detection_logitK=-0.002478460182136632,
        r_SP=3.488980445181244,
    ))
    @assert all(k -> haskey(initial, k), keys(problem.prior))

    # Keep decomposition aligned with calibration_loglikelihood.
    function components(theta; output=nothing)
        out = isnothing(output) ? TB.predict(problem, theta) : output
        ll_notifications = TB.notification_loglikelihood(
            problem.observation_model, problem.observations,
            out, problem.age_breaks, theta)
        ll_recurrence = TB.recurrence_loglikelihood(problem.recurrence, out)
        ll_site = TB.site_loglikelihood(problem.site, out)
        ll_treatment = TB.treatment_death_loglikelihood(
            problem.treatment_deaths, theta)
        return (
            notifications=ll_notifications,
            recurrence=ll_recurrence,
            site=ll_site,
            treatment_deaths=ll_treatment,
            total=ll_notifications + ll_recurrence + ll_site + ll_treatment,
        )
    end

    println("\nSIX-PARAMETER DoddExtended MLE")
    println("Free: ", join(string.(free_names), ", "))
    println("Fixed: all other medians, detection_trend=0, sigma=0.30, tau=0.18")
    println("Likelihood: notifications + recurrence + pulmonary/EP + treatment outcomes")
    println("Starts=$nstarts; max iterations per start=$iterations; solver=Tsit5")
    println("Initial parameter values:")
    for k in free_names
        println("  $k = ", getproperty(initial, k))
    end
    println("\nComputing initial likelihood (also warms up compilation)...")
    c0 = components(initial)
    println(c0)
    isfinite(c0.total) || error("Initial likelihood invalid")

    fit = TB.fit_mle(
        problem; fixed=fixed, initial=initial,
        optimizer=:nelder_mead, iterations=iterations,
        nstarts=nstarts, rng=MersenneTwister(seed))

    println("\nMLE FIT")
    println("Converged: ", fit.converged)
    println("Iterations: ", Optim.iterations(fit.optim_result))
    println("Objective evaluations: ", Optim.f_calls(fit.optim_result))
    println("Log likelihood: ", fit.log_likelihood)
    println("MAP reference log likelihood: -950.1934951044775")
    println("Log-likelihood gain over MAP: ",
        fit.log_likelihood - (-950.1934951044775))
    println("Fitted parameters:")
    for k in free_names
        println("  $k = ", getproperty(fit.parameters, k))
    end
    fit.parameters.beta > 40 && println(
        "WARNING: beta > 40; inspect force of infection and epidemic plausibility")

    c1 = components(fit.parameters)
    @assert isapprox(c1.total, fit.log_likelihood; atol=1e-4)
    println("\nLikelihood decomposition before -> after:")
    comp_df = DataFrame(
        component=String.(collect(keys(c0))),
        initial=Float64.(collect(values(c0))),
        fitted=Float64.(collect(values(c1))))
    show(stdout, MIME("text/plain"), comp_df; allrows=true)
    println()
    CSV.write(joinpath(result_dir, "likelihood_components.csv"), comp_df)

    params_df = DataFrame(
        parameter=String.(collect(keys(fit.parameters))),
        initial=Float64.(collect(values(initial))),
        fitted=Float64.(collect(values(fit.parameters))),
        free=[k in free_names for k in keys(fit.parameters)])
    CSV.write(joinpath(result_dir, "parameters.csv"), params_df)

    # Compare the likelihood components at MAP and MLE (same objective).
    # Predict over all observation years and extract epidemiological checks.
    out = TB.predict(problem, fit.parameters;
        variables=(
            :population, :incidence, :notifications, :mortality,
            :untreated_mortality, :treatment_mortality,
            :recurrent_notifications, :pulmonary_notifications,
            :extrapulmonary_notifications,
            :latent_prevalence, :untreated_active_prevalence,
            :primary_infections, :reinfections,
            :fast_progression, :slow_progression, :relapse,
        ))

    annual_rows = NamedTuple[]
    for (j, year) in enumerate(out.years)
        total(k) = sum(@view getproperty(out, k)[:, j])
        pop = total(:population)
        inc = total(:incidence)
        notif = total(:notifications)
        push!(annual_rows, (
            year=year, population=pop, incidence=inc,
            incidence_per_100k=1e5 * inc / pop,
            notifications=notif,
            tb_deaths=total(:mortality),
            mortality_per_100k=1e5 * total(:mortality) / pop,
            untreated_deaths=total(:untreated_mortality),
            treatment_deaths=total(:treatment_mortality),
            latent_pct=100 * total(:latent_prevalence) / pop,
            active_pct=100 * total(:untreated_active_prevalence) / pop,
            primary_infections=total(:primary_infections),
            reinfections=total(:reinfections),
            fast_fraction=inc > 0 ? total(:fast_progression)/inc : NaN,
            recurrent_fraction=notif > 0 ?
                               total(:recurrent_notifications)/notif : NaN,
            pulmonary_fraction=notif > 0 ?
                               total(:pulmonary_notifications)/notif : NaN,
        ))
    end
    annual = DataFrame(annual_rows)
    CSV.write(joinpath(result_dir, "annual_diagnostics.csv"), annual)

    age_fit = DataFrame(TB.notification_fit_rows(problem, out))
    CSV.write(joinpath(result_dir, "notification_fit.csv"), age_fit)

    pars = TB.make_parameters(model, fit.parameters)
    detection = DataFrame(
        year=[2013, 2019, 2023, 2024],
        reference_probability=[TB.detection_probability(pars.detection, y)
                               for y in (2013, 2019, 2023, 2024)],
        r_SP=fill(pars.relative_detection[1], 4),
        r_SN=fill(pars.relative_detection[2], 4),
        r_EP=fill(pars.relative_detection[3], 4))
    CSV.write(joinpath(result_dir, "detection_diagnostics.csv"), detection)

    println("\nEPIDEMIOLOGICAL CHECKS")
    show(stdout, MIME("text/plain"),
        filter(r -> r.year in (2013, 2019, 2023, 2024), annual);
        allrows=true)
    println("\n\n2023 AGE-SPECIFIC NOTIFICATION FIT")
    show(stdout, MIME("text/plain"),
        filter(r -> r.year == 2023, age_fit); allrows=true)
    println("\n\nSaved output: $result_dir")
    println("Comparison: MAP vs MLE (all other parameters held fixed)")
    compare_df = DataFrame(
        parameter=String.(free_names),
        MAP=[Float64(getproperty(initial, k)) for k in free_names],
        MLE=[Float64(getproperty(fit.parameters, k)) for k in free_names],
    )
    show(stdout, MIME("text/plain"), compare_df; allrows=true)
    println()
    CSV.write(joinpath(result_dir, "map_vs_mle.csv"), compare_df)
    if !fit.converged
        println("WARNING: optimiser did not report convergence; treat as provisional")
    end
end
