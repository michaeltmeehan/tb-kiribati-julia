# Six-parameter DoddExtended MCMC using Turing.jl + AdvancedMH.jl.
# Save in scripts/DoddExtended/mcmc_six_parameter_turing.jl.
# Run from repository root:
#   include("scripts/DoddExtended/mcmc_six_parameter_turing.jl")
# Optional BEFORE include, e.g. ENV["TB_MCMC_SAMPLES"] = "500":
#   TB_MCMC_CHAINS (4), TB_MCMC_WARMUP (1500), TB_MCMC_SAMPLES (2500),
#   TB_MCMC_PREDICT (200), TB_MCMC_SEED (20261125)
# This reproduces the six-parameter MAP/MLE statistical model, without WHO targets.

using TBKiribatiJulia
using Turing
using AdvancedMH
using MCMCChains
using CSV, DataFrames, Random, Statistics
using Distributions: Normal, truncated, median, params, quantile
using LinearAlgebra
using ForwardDiff
using Statistics

primal(x::ForwardDiff.Dual) = ForwardDiff.value(x)
primal(x::Real) = x

# The priors are specified ONCE here through the model's priors. The custom
# likelihood below contributes only to the log likelihood, not the prior.
@model function tb_six_parameter_posterior(problem, defaults, prior)
    beta ~ prior.beta
    Lambda0 ~ prior.Lambda0
    nu ~ prior.nu
    omega ~ prior.omega
    detection_logitK ~ prior.detection_logitK
    r_SP ~ prior.r_SP

    theta = merge(defaults, (
        beta=Float64(primal(beta)),
        Lambda0=Float64(primal(Lambda0)),
        nu=Float64(primal(nu)),
        omega=Float64(primal(omega)),
        detection_logitK=Float64(primal(detection_logitK)),
        r_SP=Float64(primal(r_SP)),
    ))
    ll = TBKiribatiJulia.calibration_loglikelihood(problem, theta)
    Turing.@addlogprob! ll
end

let
    TB = TBKiribatiJulia
    free = (:beta, :Lambda0, :nu, :omega, :detection_logitK, :r_SP)
    nchains = parse(Int, get(ENV, "TB_MCMC_CHAINS", "4"))
    nwarm = parse(Int, get(ENV, "TB_MCMC_WARMUP", "1500"))
    nsamples = parse(Int, get(ENV, "TB_MCMC_SAMPLES", "2500"))
    npredict = parse(Int, get(ENV, "TB_MCMC_PREDICT", "200"))
    seed = parse(Int, get(ENV, "TB_MCMC_SEED", "20261125"))
    nchains >= 2 || error("Use at least two chains for convergence assessment")
    nwarm > 0 && nsamples > 0 || error("Warmup and samples must be positive")
    npredict >= 0 || error("TB_MCMC_PREDICT must be nonnegative")

    output_dir = normpath(joinpath(@__DIR__, "..", "..", "results",
        "DoddExtended", "six_parameter_turing"))
    mkpath(output_dir)

    # Same setup, observation likelihood, and solver as the validated MLE script.
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

    observation_model = TB.AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=40)
    problem = TB.CalibrationProblem(
        model, setup.demography, notifications;
        start_year=1970, end_year=2026,
        age_breaks=setup.age_breaks,
        observation_model=observation_model,
        recurrence=recurrence, site=site,
        treatment_deaths=treatment_deaths,
        solver=TB.Tsit5())

    all(k -> haskey(problem.prior, k), free) ||
        error("Missing a free-parameter prior in parameters.jl")
    defaults = merge(map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (detection_trend=0.0, sigma_obs=0.30, tau_year=0.18))
    posterior = tb_six_parameter_posterior(problem, defaults, problem.prior)

    map_start = (
        beta=10.985230913865225,
        Lambda0=0.006339206566999288,
        nu=0.015186477942370179,
        omega=0.0018303349840744073,
        detection_logitK=-0.002478460182136632,
        r_SP=3.488980445181244,
    )
    mle_start = (
        beta=40.59102290208189,
        Lambda0=0.006126310967807153,
        nu=0.010180990698442803,
        omega=0.0011736289238244793,
        detection_logitK=0.9273526508798575,
        r_SP=9.920039617699409,
    )
    check = TB.calibration_loglikelihood(problem, merge(defaults, map_start))
    println("\nTURING + ADVANCEDMH: SIX-PARAMETER DoddExtended")
    println("MAP log likelihood recheck: ", check)
    isapprox(check, -950.1934951044775; atol=0.1) ||
        error("Likelihood changed from the validated MAP setup; check code/data")
    println("Chains=$nchains; warmup=$nwarm; saved samples/chain=$nsamples")
    println("Sampler: AdvancedMH.RobustAdaptiveMetropolis; no gradients")
    println("Output: ", output_dir)

    # Start chains at MAP, MLE and two slightly dispersed MAP points.
    # AdvancedMH operates through Turing's transformed-variable interface.
    starts = Vector{typeof(map_start)}(undef, nchains)
    for c in 1:nchains
        rng = MersenneTwister(seed + c)
        starts[c] = if c == 1
            map_start
        elseif c == 2
            mle_start
        else
            merge(map_start, (
                beta=map_start.beta * exp(0.15 * randn(rng)),
                Lambda0=map_start.Lambda0 * exp(0.15 * randn(rng)),
                nu=map_start.nu * exp(0.15 * randn(rng)),
                omega=map_start.omega * exp(0.15 * randn(rng)),
                detection_logitK=map_start.detection_logitK + 0.2 * randn(rng),
                r_SP=map_start.r_SP * exp(0.2 * randn(rng)),
            ))
        end
    end

    # Estimate initial proposal geometry from the previous pilot.
    pilot = CSV.read(
        joinpath(
            "results", "DoddExtended",
            "six_parameter_turing", "posterior_draws.csv"
        ),
        DataFrame
    )

    # Match Turing's unconstrained parameter coordinates.
    Z = hcat(
        log.(pilot.beta),
        log.(pilot.Lambda0),
        log.(pilot.nu),
        log.(pilot.omega),
        pilot.detection_logitK,
        log.(pilot.r_SP),
    )

    Σ = cov(Z; dims=1)

    # Cholesky factor, scaled for a six-dimensional random walk.
    S = (2.38 / sqrt(6)) *
        Matrix(cholesky(Symmetric(Σ + 1e-6I)).L)

    chain_tables = DataFrame[]
    sampled_chains = MCMCChains.Chains[]
    for c in 1:nchains
        println("\nSampling chain $c / $nchains ...")
        rng = MersenneTwister(seed + 100c)
        # In Turing's linked space the positive variables are on log scale.
        sampler = Turing.externalsampler(
            AdvancedMH.RobustAdaptiveMetropolis(S=S)
        )
        chn = Turing.sample(
            rng, posterior, sampler, nsamples;
            num_warmup=nwarm, discard_initial=nwarm,
            initial_params=Turing.InitFromParams(starts[c]),
            chain_type=MCMCChains.Chains,
            progress=true)
        push!(sampled_chains, chn)

        # Select only the six model parameters (exclude sampler internals).
        values = Array(chn[:, collect(free), :])[:, :, 1]
        df = DataFrame(values, collect(free))
        insertcols!(df, 1, :iteration => collect(1:nrow(df)))
        insertcols!(df, 1, :chain => fill(c, nrow(df)))
        CSV.write(joinpath(output_dir, "chain_$(c).csv"), df)
        push!(chain_tables, df)
        println("Saved chain $c: ", nrow(df), " posterior draws")
    end

    combined = reduce(vcat, chain_tables)
    CSV.write(joinpath(output_dir, "posterior_draws.csv"), combined)

    # --------------------------------------------------
    # Save calibration parameter specifications
    # --------------------------------------------------

    prior_table = DataFrame([
        (
            parameter=String(k),
            distribution=String(nameof(typeof(d))),
            param1=params(d)[1],
            param2=params(d)[2],
            median=median(d),
            q025=quantile(d, 0.025),
            q975=quantile(d, 0.975),
        )
        for (k, d) in pairs(problem.prior)
        if k in free
    ])

    CSV.write(
        joinpath(output_dir, "prior_parameters.csv"),
        prior_table,
    )

    fixed_table = DataFrame([
        (
            parameter=String(k),
            value=v,
        )
        for (k, v) in pairs(defaults)
        if !(k in free)
    ])

    CSV.write(
        joinpath(output_dir, "fixed_parameters.csv"),
        fixed_table,
    )

    # --------------------------------------------------
    # Save run configuration
    # --------------------------------------------------

    repo_dir = normpath(joinpath(@__DIR__, "..", ".."))

    git_commit = try
        readchomp(`git -C $repo_dir rev-parse HEAD`)
    catch
        "unknown"
    end

    git_dirty = try
        !isempty(readchomp(
            `git -C $repo_dir status --porcelain`
        ))
    catch
        missing
    end

    configuration = DataFrame(
        nchains=[nchains],
        warmup=[nwarm],
        samples_per_chain=[nsamples],
        posterior_predictions=[npredict],
        seed=[seed],
        sampler=["RobustAdaptiveMetropolis"],
        julia_version=[string(VERSION)],
        git_commit=[git_commit],
        git_dirty=[git_dirty],
    )

    CSV.write(
        joinpath(output_dir, "run_configuration.csv"),
        configuration,
    )

    # MCMCChains provides a standard chain summary (R-hat, ESS, quantiles).
    try
        all_chains = MCMCChains.chainscat(sampled_chains...)
        open(joinpath(output_dir, "mcmc_diagnostics.txt"), "w") do io
            show(io, MIME("text/plain"), MCMCChains.describe(all_chains))
            println(io)
        end
        println("\nCOMBINED CHAIN DIAGNOSTICS")
        show(stdout, MIME("text/plain"), MCMCChains.describe(all_chains))
        println()
    catch err
        @warn "Could not combine chains for diagnostics; per-chain CSVs saved" exception=(err, catch_backtrace())
    end

    # Posterior deterministic epidemiology: randomly select draws across chains.
    # This is NOT a posterior predictive replication of observed surveillance.
    if npredict > 0
        rng = MersenneTwister(seed + 999)
        selected = rand(rng, 1:nrow(combined), min(npredict, nrow(combined)))
        rows = NamedTuple[]
        for (i, ix) in enumerate(selected)
            r = combined[ix, :]
            vals = NamedTuple{free}(Tuple(Float64(r[k]) for k in free))
            theta = merge(defaults, vals)
            out = TB.predict(problem, theta; variables=(
                :population, :incidence, :notifications, :mortality,
                :latent_prevalence, :untreated_active_prevalence,
                :recurrent_notifications, :pulmonary_notifications,
            ))
            for year in (2013, 2019, 2023, 2024)
                j = findfirst(==(year), out.years)
                isnothing(j) && continue
                total(k) = sum(@view getproperty(out, k)[:, j])
                pop = total(:population)
                n = total(:notifications)
                push!(rows, (
                    draw_index=ix, chain=Int(r.chain), year=year,
                    incidence_per_100k=1e5 * total(:incidence) / pop,
                    mortality_per_100k=1e5 * total(:mortality) / pop,
                    notifications=n,
                    latent_pct=100 * total(:latent_prevalence) / pop,
                    active_pct=100 * total(:untreated_active_prevalence) / pop,
                    recurrent_fraction=n > 0 ? total(:recurrent_notifications) / n : NaN,
                    pulmonary_fraction=n > 0 ? total(:pulmonary_notifications) / n : NaN,
                ))
            end
            i % 25 == 0 && println("Posterior predictions: $i / $(length(selected))")
        end
        epidemic = DataFrame(rows)
        CSV.write(joinpath(output_dir, "posterior_epidemiology.csv"), epidemic)
        # Store flattened summary (one row per year and outcome).
        summary_rows = NamedTuple[]
        for g in groupby(epidemic, :year), variable in (
                :incidence_per_100k, :mortality_per_100k, :notifications,
                :latent_pct, :active_pct, :recurrent_fraction, :pulmonary_fraction)

            q = quantile(g[!, variable], [0.025, 0.5, 0.975])
            push!(summary_rows, (year=first(g.year), metric=String(variable),
                q025=q[1], median=q[2], q975=q[3]))
        end
        CSV.write(joinpath(output_dir, "posterior_epidemiology_summary.csv"),
            DataFrame(summary_rows))
    end
    println("\nCompleted. Saved results to ", output_dir)
end
