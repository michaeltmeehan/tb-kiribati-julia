# Posterior epidemiological predictions from the converged six-parameter MCMC.
# Save at scripts/DoddExtended/posterior_epidemiology_doddextended.jl.
# Run from repository root:
#   include("scripts/DoddExtended/posterior_epidemiology_doddextended.jl")
#
# Optional BEFORE include:
#   ENV["TB_POST_NDRAWS"] = "500"         # number of draws, stratified by chain
#   ENV["TB_POST_SEED"] = "20261125"
#   ENV["TB_POST_YEAR_START"] = "2000"   # earliest reporting year (not ODE start)
#   ENV["TB_POST_YEAR_END"] = "2025"
#
# This is deterministic epidemiological uncertainty conditional on parameter
# draws, NOT posterior predictive replication of surveillance counts.
# WHO incidence/mortality estimates are not used in model fitting or prediction.

using TBKiribatiJulia
using CSV, DataFrames, Random, Statistics
using Distributions: Normal, truncated, median

let
    TB = TBKiribatiJulia
    free = (:beta, :Lambda0, :nu, :omega, :detection_logitK, :r_SP)
    ndraws = parse(Int, get(ENV, "TB_POST_NDRAWS", "500"))
    seed = parse(Int, get(ENV, "TB_POST_SEED", "20261125"))
    report_start = parse(Int, get(ENV, "TB_POST_YEAR_START", "2000"))
    report_end = parse(Int, get(ENV, "TB_POST_YEAR_END", "2025"))
    ndraws > 0 || error("TB_POST_NDRAWS must be positive")
    report_start <= report_end || error("Invalid reporting years")

    root = normpath(joinpath(@__DIR__, "..", ".."))
    input_file = joinpath(root, "results", "DoddExtended",
        "six_parameter_turing", "posterior_draws.csv")
    outdir = joinpath(root, "results", "DoddExtended", "posterior_epidemiology")
    isfile(input_file) || error("MCMC draws not found: $input_file")
    mkpath(outdir)

    draws = CSV.read(input_file, DataFrame)
    required_cols = (:chain, free...)
    missing_cols = filter(k -> !(k in propertynames(draws)), required_cols)
    isempty(missing_cols) || error("Missing posterior columns: $missing_cols")
    all(k -> all(isfinite, draws[!, k]), free) || error("Nonfinite posterior parameters")
    all(k -> all(x -> x > 0, draws[!, k]),
        (:beta, :Lambda0, :nu, :omega, :r_SP)) ||
        error("Posterior sample contains nonpositive rate/multiplier")

    # Even allocation over the chains, random sampling WITHOUT replacement.
    # Row positions are retained for reproducibility and joint outcome analysis.
    rng = MersenneTwister(seed)
    chain_ids = sort(unique(draws.chain))
    nchain = length(chain_ids)
    nchain >= 2 || error("Expected multiple MCMC chains")
    quotient, remainder = divrem(ndraws, nchain)
    selected_rows = Int[]
    for (i, chain_id) in enumerate(chain_ids)
        ix = findall(==(chain_id), draws.chain)
        n = quotient + (i <= remainder ? 1 : 0)
        n <= length(ix) || error("Requested $n draws from chain $chain_id, but only $(length(ix)) exist")
        append!(selected_rows, ix[randperm(rng, length(ix))[1:n]])
    end
    shuffle!(rng, selected_rows)
    println("\nDoddExtended posterior epidemiology")
    println("Source: $input_file")
    println("Posterior draws: $(length(selected_rows)), across $nchain chains")
    println("Reporting years: $report_start:$report_end")

    # Identical epidemiological model, surveillance setup, fixed parameters,
    # observation model and solver as the validated six-parameter MAP/MLE.
    include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))
    setup = setup_dodd_analysis(
        start_year=1970, end_year=2026, output_years=2013:2024)
    pres = TB.load_kiribati_presentation(
        setup.surveillance_dir; years=2007:2012)
    model = TB.DoddExtendedTBModel(
        setup.contact, pres.presentation; reinfection=true)

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

    defaults = merge(map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (detection_trend=0.0, sigma_obs=0.30, tau_year=0.18))

    # Validate reproduction of the calibrated reference before processing draws.
    map_fit = merge(defaults, (
        beta=10.985230913865225,
        Lambda0=0.006339206566999288,
        nu=0.015186477942370179,
        omega=0.0018303349840744073,
        detection_logitK=-0.002478460182136632,
        r_SP=3.488980445181244,
    ))
    reference_ll = TB.calibration_loglikelihood(problem, map_fit)
    println("Reference MAP likelihood: $reference_ll")
    isapprox(reference_ll, -950.1934951044775; atol=0.1) ||
        error("The calibration setup no longer reproduces the reference MAP likelihood")

    prediction_vars = (
        :population, :incidence, :notifications, :mortality,
        :untreated_mortality, :treatment_mortality,
        :recurrent_notifications, :pulmonary_notifications,
        :extrapulmonary_notifications,
        :latent_prevalence, :untreated_active_prevalence,
        :primary_infections, :reinfections,
        :fast_progression, :slow_progression, :relapse,
    )

    # Collect rows in long format so posterior intervals can be calculated
    # separately for each outcome/year or age-group/year.
    annual_rows = NamedTuple[]
    age_rows = NamedTuple[]
    failure_rows = NamedTuple[]
    selected_parameters = NamedTuple[]

    for (number, row_id) in enumerate(selected_rows)
        r = draws[row_id, :]
        pars = NamedTuple{free}(Tuple(Float64(r[k]) for k in free))
        theta = merge(defaults, pars)
        chain_id = Int(r.chain)
        iteration_id = :iteration in propertynames(draws) ? Int(r.iteration) : row_id
        try
            pred = TB.predict(problem, theta; variables=prediction_vars)
            # Extract annual rates and prevalence. These are latent model
            # outcomes, not NB2 replicated notification counts.
            for (j, year) in enumerate(pred.years)
                report_start <= year <= report_end || continue
                total(k) = sum(@view getproperty(pred, k)[:, j])
                population = total(:population)
                incidence = total(:incidence)
                notif = total(:notifications)
                deaths = total(:mortality)
                population > 0 || error("Nonpositive population in year $year")
                push!(annual_rows, (
                    draw_id=number, source_row=row_id,
                    chain=chain_id, iteration=iteration_id, year=year,
                    population=population,
                    incidence=incidence,
                    incidence_per_100k=1e5 * incidence/population,
                    notifications=notif,
                    tb_deaths=deaths,
                    mortality_per_100k=1e5 * deaths/population,
                    untreated_deaths=total(:untreated_mortality),
                    treatment_deaths=total(:treatment_mortality),
                    latent_pct=100 * total(:latent_prevalence)/population,
                    active_pct=100 * total(:untreated_active_prevalence)/population,
                    recurrent_fraction=notif > 0 ?
                                       total(:recurrent_notifications)/notif : NaN,
                    pulmonary_fraction=notif > 0 ?
                                       total(:pulmonary_notifications)/notif : NaN,
                    beta=pars.beta, Lambda0=pars.Lambda0,
                    nu=pars.nu, omega=pars.omega,
                    detection_logitK=pars.detection_logitK,
                    r_SP=pars.r_SP,
                ))
            end

            # Existing utility returns EXACT surveillance age groups, including
            # the combined 0-14 category in 2024. No invented age rebinning.
            age_fit = TB.notification_fit_rows(problem, pred)
            for a in age_fit
                report_start <= a.year <= report_end || continue
                a.age_group == "all" && continue
                push!(age_rows, (
                    draw_id=number, source_row=row_id,
                    chain=chain_id, year=Int(a.year),
                    age_group=String(a.age_group),
                    observed=Float64(a.observed),
                    expected=Float64(a.expected),
                ))
            end
            push!(selected_parameters, merge((
                draw_id=number, source_row=row_id, chain=chain_id,
                iteration=iteration_id), pars))
        catch err
            push!(failure_rows, (
                draw_id=number, source_row=row_id,
                chain=chain_id, error=sprint(showerror, err)))
            @warn "Posterior prediction failed for draw $number" error=sprint(showerror, err)
        end
        if number % 25 == 0 || number == length(selected_rows)
            println("Processed $number / $(length(selected_rows)) draws")
        end
    end

    CSV.write(joinpath(outdir, "failed_draws.csv"),
        isempty(failure_rows) ? DataFrame(draw_id=Int[], source_row=Int[], chain=Int[], error=String[]) : DataFrame(failure_rows))
    isempty(selected_parameters) && error("All posterior prediction evaluations failed")
    if !isempty(failure_rows)
        @warn "Posterior predictions failed for $(length(failure_rows)) draws; inspect failed_draws.csv. Missing draws can bias intervals."
    end

    annual = DataFrame(annual_rows)
    age = DataFrame(age_rows)
    CSV.write(joinpath(outdir, "selected_draws.csv"), DataFrame(selected_parameters))
    CSV.write(joinpath(outdir, "annual_draws.csv"), annual)
    CSV.write(joinpath(outdir, "age_notification_draws.csv"), age)

    # Quantile summaries of posterior parameter uncertainty propagated through
    # the deterministic system. No stochastic surveillance NB2 noise is added.
    annual_metrics = (
        :incidence, :incidence_per_100k, :notifications,
        :tb_deaths, :mortality_per_100k,
        :untreated_deaths, :treatment_deaths,
        :latent_pct, :active_pct,
        :recurrent_fraction, :pulmonary_fraction,
    )
    annual_summary = NamedTuple[]
    for group in groupby(annual, :year), metric in annual_metrics
        vals = filter(isfinite, Float64.(group[!, metric]))
        isempty(vals) && continue
        q = quantile(vals, [0.025, 0.5, 0.975])
        push!(annual_summary, (
            year=Int(first(group.year)), metric=String(metric),
            n=length(vals), q025=q[1], median=q[2], q975=q[3],
        ))
    end
    CSV.write(joinpath(outdir, "annual_summary.csv"),
        DataFrame(annual_summary))

    age_summary = NamedTuple[]
    if nrow(age) > 0
        for group in groupby(age, [:year, :age_group])
            q = quantile(group.expected, [0.025, 0.5, 0.975])
            push!(age_summary, (
                year=Int(first(group.year)),
                age_group=String(first(group.age_group)),
                observed=Float64(first(group.observed)),
                n=nrow(group), q025=q[1], median=q[2], q975=q[3],
            ))
        end
    end
    CSV.write(joinpath(outdir, "age_notification_summary.csv"),
        isempty(age_summary) ? DataFrame(year=Int[], age_group=String[], observed=Float64[], n=Int[], q025=Float64[], median=Float64[], q975=Float64[]) : DataFrame(age_summary))

    println("\n2023 posterior epidemiological predictions")
    summary_df = DataFrame(annual_summary)
    preview = filter(r -> r.year == 2023 && r.metric in (
        "incidence_per_100k", "notifications", "mortality_per_100k",
        "latent_pct", "active_pct", "recurrent_fraction",
        "pulmonary_fraction"), summary_df)
    show(stdout, MIME("text/plain"), preview; allrows=true)
    println("\n\nSaved to: $outdir")
    println("Files: selected_draws.csv, annual_draws.csv, annual_summary.csv,")
    println("       age_notification_draws.csv, age_notification_summary.csv,")
    println("       failed_draws.csv")
    println("Intervals quantify parameter uncertainty conditional on fixed natural history and model structure.")
end
