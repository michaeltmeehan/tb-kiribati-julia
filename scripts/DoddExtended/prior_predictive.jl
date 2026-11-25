# Standalone conditional prior-predictive analysis for DoddExtended.
# Place at scripts/DoddExtended/prior_predictive.jl, run from repo root:
#   include("scripts/DoddExtended/prior_predictive.jl")
# Optional in PowerShell: $env:TB_PRIOR_DRAWS="1000"
# Uses current natural-history priors for the six calibrated parameters;
# all other epidemiological parameters are FIXED at their prior medians.
# Observation dispersion is FIXED at sigma_obs=0.30, tau_year=0.18.

using TBKiribatiJulia
using CSV, DataFrames, Random, Statistics
using Distributions: Normal, NegativeBinomial, median, truncated
using SciMLBase: successful_retcode

let
    TB = TBKiribatiJulia
    N = parse(Int, get(ENV, "TB_PRIOR_DRAWS", "500"))
    seed = parse(Int, get(ENV, "TB_PRIOR_SEED", "20261125"))
    rng = MersenneTwister(seed)
    years = collect(2013:2024)
    output_dir = normpath(joinpath(@__DIR__, "..", "..", "results",
        "DoddExtended", "prior_predictive"))
    mkpath(output_dir)

    include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))
    setup = setup_dodd_analysis(start_year=1970, end_year=2026,
        output_years=2013:2024)
    presentation = TB.load_kiribati_presentation(
        setup.surveillance_dir; years=2007:2012)
    model = TB.DoddExtendedTBModel(
        setup.contact, presentation.presentation; reinfection=true)

    national = filter(r -> 1980 <= r.year <= 2025,
        setup.surveillance.national_notifications)
    observations = TB.combine_notification_observations(
        setup.age_observations, national;
        n_age_groups=length(setup.age_labels),
        start_year=1970, end_year=2026)
    observation_model = TB.AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=30)
    problem = TB.CalibrationProblem(
        model, setup.demography, observations;
        start_year=1970, end_year=2026,
        age_breaks=setup.age_breaks,
        observation_model=observation_model,
        solver=TB.Tsit5())

    # The intended calibration parameter set. No detection trend is sampled.
    free_names = (:beta, :Lambda0, :nu, :omega,
        :detection_logitK, :r_SP)
    missing = filter(k -> !haskey(problem.prior, k), free_names)
    isempty(missing) || error("Missing priors for $(missing). Check parameters.jl")

    fixed = merge(map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (detection_trend=0.0, sigma_obs=0.30, tau_year=0.18))
    println("\nDoddExtended conditional prior predictive: $N draws; seed=$seed")
    println("Free: ", join(string.(free_names), ", "))
    println("Fixed: other natural-history medians, trend=0, sigma=0.30, tau=0.18")
    println("No optimisation. Full deterministic epidemic simulations.")

    vars = (:population, :incidence, :notifications, :mortality,
        :recurrent_notifications, :primary_infections, :reinfections,
        :fast_progression, :slow_progression, :relapse,
        :pulmonary_notifications, :extrapulmonary_notifications,
        :untreated_mortality, :treatment_mortality,
        :latent_prevalence, :untreated_active_prevalence)
    parameter_rows = NamedTuple[]
    annual_rows = NamedTuple[]
    age_rows = NamedTuple[]
    errors = NamedTuple[]

    function simulate_counts!(rng, output, yi, sigma, tau)
        # One common, mean-one lognormal observation effect for the YEAR.
        annual_multiplier = exp(tau * randn(rng) - tau^2 / 2)
        phi = 1 / sigma^2
        noisy = zeros(Int, size(output.notifications, 1))
        for g in eachindex(noisy)
            mu = max(0.0, output.notifications[g, yi] * annual_multiplier)
            noisy[g] = mu == 0 ? 0 : rand(rng,
                NegativeBinomial(phi, phi / (phi + mu)))
        end
        return noisy
    end

    for i in 1:N
        # Sample the JOINT prior only for the six free parameters.
        sampled = NamedTuple{free_names}(
            Tuple(rand(rng, getproperty(problem.prior, k)) for k in free_names))
        theta = merge(fixed, sampled)
        push!(parameter_rows, merge((draw=i,), sampled))
        try
            sol = TB.simulate_from_draw(problem.simulation,
                TB.model_draw(problem, theta))
            successful_retcode(sol) || error("ODE retcode: $(sol.retcode)")
            out = TB.model_outputs(model, sol;
                variables=vars, years=years,
                age_breaks=setup.age_breaks)
            ng, ny = size(out.notifications)
            ng == length(setup.age_labels) || error("Unexpected group count: $ng")
            ny == length(years) || error("Unexpected year count: $ny")

            for (j, year) in enumerate(years)
                sumvar(s) = sum(@view getproperty(out, s)[:, j])
                pop = sumvar(:population)
                inc = sumvar(:incidence)
                ntf = sumvar(:notifications)
                noisy = simulate_counts!(rng, out, j,
                    theta.sigma_obs, theta.tau_year)
                push!(annual_rows, (
                    draw=i, year=year, population=pop,
                    incidence=inc, incidence_per_100k=1e5*inc/pop,
                    notifications_mean=ntf,
                    notifications_predictive=sum(noisy),
                    mortality=sumvar(:mortality),
                    mortality_per_100k=1e5*sumvar(:mortality)/pop,
                    untreated_deaths=sumvar(:untreated_mortality),
                    treatment_deaths=sumvar(:treatment_mortality),
                    recurrent_fraction=ntf>0 ?
                                       sumvar(:recurrent_notifications)/ntf : NaN,
                    pulmonary_fraction=ntf>0 ?
                                       sumvar(:pulmonary_notifications)/ntf : NaN,
                    latent_pct=100*sumvar(:latent_prevalence)/pop,
                    active_pct=100*sumvar(:untreated_active_prevalence)/pop,
                    primary_infections=sumvar(:primary_infections),
                    reinfections=sumvar(:reinfections),
                    fast_fraction=inc>0 ? sumvar(:fast_progression)/inc : NaN,
                ))
                if year == 2023
                    for g in 1:ng
                        push!(age_rows, (
                            draw=i, year=year,
                            age_group=setup.age_labels[g],
                            notifications_mean=out.notifications[g, j],
                            notifications_predictive=noisy[g],
                            incidence=out.incidence[g, j],
                            population=out.population[g, j],
                        ))
                    end
                end
            end
        catch err
            push!(errors, (draw=i, reason=sprint(showerror, err)))
        end
        i % 50 == 0 && println("Processed $i/$N; failed $(length(errors))")
    end

    params_df = DataFrame(parameter_rows)
    annual_df = DataFrame(annual_rows)
    age_df = DataFrame(age_rows)
    failure_df = DataFrame(draw=Int[], reason=String[])
    for e in errors
        push!(failure_df, e)
    end
    CSV.write(joinpath(output_dir, "draws_parameters.csv"), params_df)
    CSV.write(joinpath(output_dir, "draws_annual.csv"), annual_df)
    CSV.write(joinpath(output_dir, "draws_age_2023.csv"), age_df)
    CSV.write(joinpath(output_dir, "failures.csv"), failure_df)

    isempty(annual_rows) && error("All simulations failed; see failures.csv")
    metrics = (:incidence_per_100k, :notifications_mean,
        :notifications_predictive, :mortality_per_100k,
        :latent_pct, :active_pct, :recurrent_fraction,
        :pulmonary_fraction, :reinfections, :fast_fraction)
    summaries = NamedTuple[]
    for yr in (2013, 2019, 2023, 2024)
        part = annual_df[annual_df.year .== yr, :]
        for m in metrics
            v = filter(isfinite, Float64.(part[!, m]))
            isempty(v) && continue
            q = quantile(v, [0.025, 0.25, 0.5, 0.75, 0.975])
            push!(summaries, (year=yr, metric=String(m),
                n=length(v), q025=q[1], q25=q[2],
                median=q[3], q75=q[4], q975=q[5]))
        end
    end
    summary_df = DataFrame(summaries)
    CSV.write(joinpath(output_dir, "prior_predictive_summary.csv"), summary_df)

    # Screening-level global sensitivity: Spearman rank correlation between
    # each sampled parameter and each 2023 outcome. Not a Sobol index.
    function ranks(x::AbstractVector)
        n = length(x)
        idx = sortperm(x)
        r = zeros(Float64, n)
        for (j, ix) in enumerate(idx)
            r[ix] = j
        end
        return r
    end
    sensitivity = NamedTuple[]
    yr2023 = annual_df[annual_df.year .== 2023, :]
    for parameter in free_names, metric in metrics
        joined = innerjoin(select(params_df, :draw, parameter),
            select(yr2023, :draw, metric), on=:draw)
        a = Float64.(joined[!, parameter])
        b = Float64.(joined[!, metric])
        good = isfinite.(a) .& isfinite.(b)
        a, b = a[good], b[good]
        if length(a) > 2 && std(a) > 0 && std(b) > 0
            push!(sensitivity, (parameter=String(parameter),
                metric=String(metric), n=length(a),
                spearman=cor(ranks(a), ranks(b))))
        end
    end
    CSV.write(joinpath(output_dir, "rank_sensitivity_2023.csv"),
        DataFrame(sensitivity))

    println("\nCompleted: $(N-length(errors))/$N successful draws")
    println("\n2023 conditional prior predictive quantiles:")
    show(stdout, MIME("text/plain"),
        summary_df[summary_df.year .== 2023, :]; allrows=true)
    println("\n\nSaved: $output_dir")
    println("NOTE: These are conditional prior predictives: only six epidemiological")
    println("parameters vary; other natural-history and observation parameters are fixed.")
end
