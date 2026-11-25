# Standalone profile likelihood for r_SP in DoddExtended.
# Save as scripts/DoddExtended/profile_sp_detection_likelihood.jl
# Run from repository root:
#   include("scripts/DoddExtended/profile_sp_detection_likelihood.jl")
# Optional before starting Julia:
#   ENV["TB_PROFILE_ITERATIONS"] = "1000"
#   ENV["TB_PROFILE_GRID"] = "9.920039617699409,6,4,3,2,1,12,16"
# Every grid point fixes r_SP and maximises the SAME surveillance likelihood
# over beta, Lambda0, nu, omega, detection_logitK. No priors in objective.

using TBKiribatiJulia
using CSV, DataFrames, Random
using Distributions: Normal, median, truncated
using Optim

let
    TB = TBKiribatiJulia
    start_year, end_year = 1970, 2026
    iterations = parse(Int, get(ENV, "TB_PROFILE_ITERATIONS", "1000"))
    grid = parse.(Float64, strip.(split(get(
        ENV, "TB_PROFILE_GRID", "9.920039617699409,6,4,3,2,1,12,16"), ",")))
    all(isfinite.(grid) .& (grid .> 0)) || error("r_SP grid must be finite and positive")
    length(unique(grid)) == length(grid) || error("Duplicate r_SP values in grid")
    result_dir = normpath(joinpath(@__DIR__, "..", "..", "results",
        "DoddExtended", "profile_sp_detection"))
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
    problem = TB.CalibrationProblem(
        model, setup.demography, notifications;
        start_year=start_year, end_year=end_year,
        age_breaks=setup.age_breaks,
        observation_model=obs_model,
        recurrence=recurrence, site=site,
        treatment_deaths=treatment_deaths,
        solver=TB.Tsit5())

    all_free = (:beta, :Lambda0, :nu, :omega,
        :detection_logitK, :r_SP)
    free_when_profiled = (:beta, :Lambda0, :nu, :omega, :detection_logitK)
    @assert all(k -> haskey(problem.prior, k), all_free)

    defaults = merge(map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (detection_trend=0.0,
            sigma_obs=0.30, tau_year=0.18))
    fixed_names = Tuple(k for k in keys(problem.prior) if !(k in all_free))
    base_fixed = NamedTuple{fixed_names}(
        Tuple(getproperty(defaults, k) for k in fixed_names))

    # MLE found in the previous six-parameter fit; serves ONLY as a warm start.
    mle_reference = (
        beta=40.59102290208189,
        Lambda0=0.006126310967807153,
        nu=0.010180990698442803,
        omega=0.0011736289238244793,
        detection_logitK=0.9273526508798575,
        r_SP=9.920039617699409,
    )
    reference_loglik = -948.6957176478281

    function components(theta)
        out = TB.predict(problem, theta)
        ll_notifications = TB.notification_loglikelihood(
            problem.observation_model, problem.observations,
            out, problem.age_breaks, theta)
        ll_recurrence = TB.recurrence_loglikelihood(problem.recurrence, out)
        ll_site = TB.site_loglikelihood(problem.site, out)
        ll_treatment = TB.treatment_death_loglikelihood(
            problem.treatment_deaths, theta)
        return (notifications=ll_notifications, recurrence=ll_recurrence,
            site=ll_site, treatment_deaths=ll_treatment,
            total=ll_notifications + ll_recurrence + ll_site + ll_treatment)
    end

    function burden_2023(theta)
        out = TB.predict(problem, theta;
            variables=(:population, :incidence, :notifications,
                :mortality, :latent_prevalence,
                :untreated_active_prevalence,
                :recurrent_notifications, :pulmonary_notifications))
        j = findfirst(==(2023), out.years)
        isnothing(j) && error("2023 absent from prediction output")
        total(k) = sum(@view getproperty(out, k)[:, j])
        pop = total(:population)
        inc = total(:incidence)
        notif = total(:notifications)
        return (
            incidence_per_100k=1e5 * inc / pop,
            mortality_per_100k=1e5 * total(:mortality) / pop,
            notifications_2023=notif,
            latent_pct=100 * total(:latent_prevalence) / pop,
            active_pct=100 * total(:untreated_active_prevalence) / pop,
            recurrence_fraction=notif > 0 ?
                                total(:recurrent_notifications) / notif : NaN,
            pulmonary_fraction=notif > 0 ?
                               total(:pulmonary_notifications) / notif : NaN,
        )
    end

    println("\nDoddExtended profile likelihood for r_SP")
    println("Grid (ordered for warm starts): ", grid)
    println("Five parameters reoptimised at each point: ", free_when_profiled)
    println("Likelihood: notifications + recurrence + site + treatment deaths")
    println("Fixed: other prior medians, trend=0, sigma=0.30, tau=0.18")
    println("Iterations: $iterations; one start per grid point; solver Tsit5")
    println("Output: $result_dir")

    # Start each point from its nearest successfully completed neighbour on
    # the r_SP axis (not necessarily the previously executed point).
    completed = Dict{Float64,NamedTuple}()
    rows = NamedTuple[]
    for (i, r) in enumerate(grid)
        nearest = isempty(completed) ? nothing :
                  first(sort(collect(keys(completed)); by=x -> abs(log(r/x))))
        warm = isnothing(nearest) ? mle_reference : completed[nearest]
        initial = merge(defaults, warm, (r_SP=r,))
        fixed = merge(base_fixed, (r_SP=r,))

        println("\n[$i/$(length(grid))] Fixed r_SP=$r; ",
            isnothing(nearest) ? "starting at six-parameter MLE" :
            "warm start from r_SP=$nearest")
        try
            fit = TB.fit_mle(
                problem; fixed=fixed, initial=initial,
                optimizer=:nelder_mead,
                iterations=iterations, nstarts=1,
                rng=MersenneTwister(20261125 + i))
            theta = fit.parameters
            abs(theta.r_SP - r) <= 1e-8 * max(1.0, r) ||
                error("Profiled r_SP not fixed: expected $r, got $(theta.r_SP)")
            ll = components(theta)
            isapprox(ll.total, fit.log_likelihood; atol=1e-3) ||
                error("Likelihood decomposition disagrees with fit")
            epi = burden_2023(theta)
            completed[r] = theta

            record = merge((r_SP_fixed=r, status="ok",
                converged=fit.converged,
                iterations=Optim.iterations(fit.optim_result),
                evaluations=Optim.f_calls(fit.optim_result),
                log_likelihood=fit.log_likelihood,
                ll_notifications=ll.notifications,
                ll_recurrence=ll.recurrence,
                ll_site=ll.site,
                ll_treatment_deaths=ll.treatment_deaths),
                NamedTuple{free_when_profiled}(
                    Tuple(getproperty(theta, k) for k in free_when_profiled)),
                epi)
            push!(rows, record)
            println("  converged=$(fit.converged), iterations=$(record.iterations), ",
                "loglik=$(round(fit.log_likelihood; digits=5))")
            println("  beta=$(round(theta.beta;digits=4)), ",
                "nu=$(round(theta.nu;digits=6)), ",
                "incidence=$(round(epi.incidence_per_100k;digits=1))/100k, ",
                "mortality=$(round(epi.mortality_per_100k;digits=1))/100k")
            fit.converged || println("  NOTE: not formally converged; consider rerun")
        catch err
            println("  FAILED: ", sprint(showerror, err))
            push!(rows, (r_SP_fixed=r, status="failed",
                converged=false, iterations=missing,
                evaluations=missing, log_likelihood=missing,
                ll_notifications=missing, ll_recurrence=missing,
                ll_site=missing, ll_treatment_deaths=missing,
                beta=missing, Lambda0=missing, nu=missing,
                omega=missing, detection_logitK=missing,
                incidence_per_100k=missing,
                mortality_per_100k=missing,
                notifications_2023=missing, latent_pct=missing,
                active_pct=missing, recurrence_fraction=missing,
                pulmonary_fraction=missing))
        end

        # Checkpoint after every point; safe to inspect after interruption.
        df = DataFrame(rows)
        best = maximum(skipmissing(df.log_likelihood); init=(-Inf))
        df[!, :likelihood_loss_2x] = [
            ismissing(v) ? missing : 2 * (best - v)
            for v in df.log_likelihood
        ]
        df[!, :loss_vs_original_mle_2x] = [
            ismissing(v) ? missing : 2 * (reference_loglik - v)
            for v in df.log_likelihood
        ]
        CSV.write(joinpath(result_dir, "profile_r_sp.csv"), df)
    end

    df = CSV.read(joinpath(result_dir, "profile_r_sp.csv"), DataFrame)
    sort!(df, :r_SP_fixed)
    CSV.write(joinpath(result_dir, "profile_r_sp.csv"), df)
    println("\nPROFILE RESULTS (sorted by r_SP)")
    show(stdout, MIME("text/plain"), select(df,
        :r_SP_fixed, :converged, :log_likelihood,
        :loss_vs_original_mle_2x, :beta, :nu,
        :incidence_per_100k, :mortality_per_100k);
        allrows=true)
    println("\n\nSaved: ", joinpath(result_dir, "profile_r_sp.csv"))
    println("Reference six-parameter MLE loglik = $reference_loglik")
    println("Loss columns are diagnostic; no χ² cutoff is automatically imposed.")
end
