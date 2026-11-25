# Calibrated DoddExtended demographic projections, Kiribati, 2025–2050.
# Save as scripts/projections/calibrated.jl and run from repository root:
#   include("scripts/projections/calibrated.jl")
# For posterior uncertainty, set before include:
#   ENV["TB_PROJ_NDRAWS"] = "100"  # additional MCMC draws, sampled across chains
#   ENV["TB_PROJ_SEED"] = "20261125"
# MAP always runs, even when TB_PROJ_NDRAWS == 0.
# WPP demographic inputs are deterministic in this analysis.

using TBKiribatiJulia
using CSV, DataFrames, Random, Statistics
using Distributions: Normal, truncated, median
using SciMLBase: successful_retcode

let
    TB = TBKiribatiJulia
    root = normpath(joinpath(@__DIR__, "..", ".."))
    outdir = joinpath(root, "results", "projections", "calibrated")
    mkpath(outdir)

    start_year, end_year = 2025, 2050
    years = collect(start_year:end_year)
    ndraws = parse(Int, get(ENV, "TB_PROJ_NDRAWS", "0"))
    seed = parse(Int, get(ENV, "TB_PROJ_SEED", "20261125"))
    ndraws >= 0 || error("TB_PROJ_NDRAWS must be nonnegative")
    free = (:beta, :Lambda0, :nu, :omega, :detection_logitK, :r_SP)

    # Reproduce the calibrated model and historical demographic inputs.
    include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))
    setup = setup_dodd_analysis(
        start_year=1970, end_year=2026, output_years=2013:2024)
    presentation = TB.load_kiribati_presentation(
        setup.surveillance_dir; years=2007:2012)
    model = TB.DoddExtendedTBModel(
        setup.contact, presentation.presentation; reinfection=true)

    national_obs = filter(r -> 1980 <= r.year <= 2025,
        setup.surveillance.national_notifications)
    notification_obs = TB.combine_notification_observations(
        setup.age_observations, national_obs;
        n_age_groups=length(setup.age_labels),
        start_year=1970, end_year=2026)
    recurrence = TB.RecurrentTBObservation(
        filter(r -> 1996 <= r.year <= 2024,
            setup.surveillance.recurrent_notifications), national_obs)
    site = TB.SiteTBObservation(filter(r -> 1970 <= r.year < 2026,
        setup.surveillance.site_notifications))
    treatment_deaths = TB.TreatmentDeathObservation(filter(r ->
        2012 <= r.year <= 2023, setup.surveillance.treatment_outcomes))
    obs_model = TB.AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=40)
    problem = TB.CalibrationProblem(
        model, setup.demography, notification_obs;
        start_year=1970, end_year=2026,
        age_breaks=setup.age_breaks,
        observation_model=obs_model,
        recurrence=recurrence, site=site,
        treatment_deaths=treatment_deaths, solver=TB.Tsit5())

    defaults = merge(map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (detection_trend=0.0, sigma_obs=0.30, tau_year=0.18))
    map_draw = (
        beta=10.985230913865225,
        Lambda0=0.006339206566999288,
        nu=0.015186477942370179,
        omega=0.0018303349840744073,
        detection_logitK=-0.002478460182136632,
        r_SP=3.488980445181244,
    )
    reference_ll = TB.calibration_loglikelihood(
        problem, merge(defaults, map_draw))
    println("Reference MAP log likelihood: ", reference_ll)
    isapprox(reference_ll, -950.1934951044775; atol=0.1) ||
        error("Calibration/model inputs differ from reference MAP")

    # Historical trajectory ends at 2025 AFTER the 2024->2025
    # demographic reconciliation; its final state is 2025+.
    historical_setup = TB.prepare_simulation(
        model, setup.demography;
        start_year=1970, end_year=start_year, solver=TB.Tsit5())

    draws = [(id=0, chain=0, iteration=0, theta=map_draw)]
    if ndraws > 0
        input_file = joinpath(root, "results", "DoddExtended",
            "six_parameter_turing", "posterior_draws.csv")
        isfile(input_file) || error("Missing MCMC draws: $input_file")
        posterior = CSV.read(input_file, DataFrame)
        all(k -> k in propertynames(posterior), (:chain, free...)) ||
            error("MCMC file lacks required parameter/chain columns")
        rng = MersenneTwister(seed)
        chains = sort(unique(posterior.chain))
        q, rem = divrem(ndraws, length(chains))
        selected = Int[]
        for (j, chain) in enumerate(chains)
            idx = findall(==(chain), posterior.chain)
            n = q + (j <= rem ? 1 : 0)
            n <= length(idx) || error("Not enough samples in chain $chain")
            append!(selected, idx[randperm(rng, length(idx))[1:n]])
        end
        shuffle!(rng, selected)
        for (j, i) in enumerate(selected)
            r = posterior[i, :]
            vals = NamedTuple{free}(Tuple(Float64(r[k]) for k in free))
            iteration = :iteration in propertynames(posterior) ? Int(r.iteration) : i
            push!(draws, (id=j, chain=Int(r.chain), iteration=iteration, theta=vals))
        end
    end

    national_tables = DataFrame[]
    effect_tables = DataFrame[]
    cumulative_tables = DataFrame[]
    cumulative_effect_tables = DataFrame[]
    metadata = NamedTuple[]
    failure_rows = NamedTuple[]

    # Cumulative contrasts use the exact same paired projection draws.
    function cumulative_burden(national::DataFrame)
        df = combine(groupby(national,
            [:scenario, :parameter_draw, :demographic_draw]),
            :incidence => sum => :incidence,
            :notifications => sum => :notifications,
            :mortality => sum => :mortality)
        df.year = fill(end_year, nrow(df))
        return df
    end

    for (j, d) in enumerate(draws)
        try
            theta = merge(defaults, d.theta)
            params = TB.make_parameters(model, TB.model_draw(problem, theta))
            hist = TB.simulate(historical_setup, params)
            successful_retcode(hist) || error("Historical ODE failed: $(hist.retcode)")
            isapprox(hist.t[end], Float64(start_year); atol=1e-7) ||
                error("Historical solution does not terminate at 2025")
            u0 = copy(hist.u[end]) # POST-callback state at 2025
            isapprox(sum(TB.model_output(model, hist, :population;
                years=[2024])[:, 1]),
                sum(TB.get_population(setup.demography, 2024)); rtol=1e-6) ||
                error("Historical 2024 population mismatch")

            # Living population at 2025+ must match the WPP 2025 age vector.
            layout = TB.state_layout(model)
            stride = TB.nstates(layout)
            living = [sum(u0[(age-1)*stride+c] for c in 1:layout.nepi)
                      for age in 1:layout.nages]
            target = TB.get_population(setup.demography, start_year)
            isapprox(living, target; rtol=1e-6, atol=1e-6) ||
                error("2025 historical state does not match WPP population")

            results = TB.project_scenarios(model, params,
                setup.demography.data;
                u0=u0, start_year=start_year, end_year=end_year,
                solver=TB.Tsit5())
            for (scenario, result) in pairs(results)
                isapprox(result.solution.u[1], u0; rtol=1e-10, atol=1e-8) ||
                    error("Initial states not identical: $scenario")
            end
            national = TB.summarise_projections(results, years;
                parameter_draw=d.id, demographic_draw=1)
            effects = TB.compare_projections(national)
            cumulative = cumulative_burden(national)
            cumulative_effects = TB.compare_projections(cumulative;
                variables=(:incidence, :notifications, :mortality))
            push!(national_tables, national)
            push!(effect_tables, effects)
            push!(cumulative_tables, cumulative)
            push!(cumulative_effect_tables, cumulative_effects)
            push!(metadata, merge((parameter_draw=d.id, chain=d.chain,
                iteration=d.iteration), d.theta))

            # Save full single-year age profiles for the reference MAP.
            if d.id == 0
                age_rows = NamedTuple[]
                for (scenario, result) in pairs(results)
                    o = result.outputs
                    for (yi, year) in enumerate(years), age in 0:(layout.nages-1)
                        a = age+1
                        push!(age_rows, (
                            scenario=String(scenario), year=year, age=age,
                            population=o.population[a, yi],
                            incidence=o.incidence[a, yi],
                            notifications=o.notifications[a, yi],
                            mortality=o.mortality[a, yi],
                            latent_prevalence=o.latent_prevalence[a, yi],
                            active_prevalence=o.active_prevalence[a, yi],
                        ))
                    end
                end
                CSV.write(joinpath(outdir, "map_age_specific.csv"),
                    DataFrame(age_rows))
                CSV.write(joinpath(outdir, "map_annual.csv"), national)
                CSV.write(joinpath(outdir, "map_effects.csv"), effects)
                CSV.write(joinpath(outdir, "map_cumulative.csv"), cumulative)
                CSV.write(joinpath(outdir, "map_cumulative_effects.csv"),
                    cumulative_effects)
                println("\nMAP INCIDENCE CONTRASTS (selected years):")
                display(select(filter(r -> r.year in (2030, 2040, 2050) &&
                    r.variable == :incidence, effects),
                    :year, :effect, :numerator, :denominator,
                    :difference, :percentage_change))
                println("\n2050 MAP PROJECTIONS:")
                display(filter(r -> r.year == 2050, national))
            end
        catch err
            push!(failure_rows, (parameter_draw=d.id, chain=d.chain,
                error=sprint(showerror, err)))
            @warn "Projection failed" parameter_draw=d.id exception=(err, catch_backtrace())
            d.id == 0 && rethrow(err) # Never hide a broken MAP setup.
        end
        (j % 25 == 0 || j == length(draws)) &&
            println("Processed $j / $(length(draws)) parameter sets")
    end

    CSV.write(joinpath(outdir, "failed_draws.csv"),
        isempty(failure_rows) ? DataFrame(parameter_draw=Int[], chain=Int[],
            error=String[]) : DataFrame(failure_rows))
    CSV.write(joinpath(outdir, "selected_parameters.csv"), DataFrame(metadata))
    full_national = reduce(vcat, national_tables)
    full_effects = reduce(vcat, effect_tables)
    full_cumulative = reduce(vcat, cumulative_tables)
    full_cumulative_effects = reduce(vcat, cumulative_effect_tables)
    CSV.write(joinpath(outdir, "annual_draws.csv"), full_national)
    CSV.write(joinpath(outdir, "effects_draws.csv"), full_effects)
    CSV.write(joinpath(outdir, "cumulative_draws.csv"), full_cumulative)
    CSV.write(joinpath(outdir, "cumulative_effects_draws.csv"),
        full_cumulative_effects)

    if ndraws > 0
        # Exclude MAP (draw 0) from posterior summaries.
        function intervals(df, group_cols, value_cols)
            posterior = filter(:parameter_draw => !=(0), df)
            rows = NamedTuple[]
            for g in groupby(posterior, group_cols), variable in value_cols
                vals = collect(skipmissing(g[!, variable]))
                isempty(vals) && continue
                push!(rows, merge(NamedTuple{Tuple(group_cols)}(
                    Tuple(g[1, k] for k in group_cols)), (
                    statistic=String(variable),
                    median=median(vals),
                    lower=quantile(vals, 0.025),
                    upper=quantile(vals, 0.975),
                    n=length(vals),
                )))
            end
            return DataFrame(rows)
        end
        CSV.write(joinpath(outdir, "annual_intervals.csv"),
            intervals(full_national, [:scenario, :year],
                [:population, :proportion_65_plus, :incidence,
                    :notifications, :mortality, :incidence_rate,
                    :notification_rate, :mortality_rate,
                    :latent_prevalence, :active_prevalence]))
        CSV.write(joinpath(outdir, "effects_intervals.csv"),
            intervals(full_effects, [:effect, :year, :variable],
                [:difference, :ratio, :percentage_change]))
        CSV.write(joinpath(outdir, "cumulative_intervals.csv"),
            intervals(full_cumulative, [:scenario, :year],
                [:incidence, :notifications, :mortality]))
        CSV.write(joinpath(outdir, "cumulative_effects_intervals.csv"),
            intervals(full_cumulative_effects,
                [:effect, :year, :variable],
                [:difference, :ratio, :percentage_change]))
    end
    println("\nSaved calibrated projections to: ", outdir)
    println("Posterior samples: ", ndraws,
        " requested; ", length(metadata)-1, " successful")
    println("Demographic uncertainties are not yet included.")
end
