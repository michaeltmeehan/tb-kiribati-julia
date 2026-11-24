using TBKiribatiJulia
using CSV, DataFrames
using Statistics: mean
using Distributions: Normal, truncated, median
using SciMLBase: successful_retcode

# Compare SP detection scenarios against national, age and presentation surveillance.
# Controlled full-model simulations: increase SP detection only.
# Place in scripts/DoddExtended/ and run:
#   include("scripts/DoddExtended/test_sp_detection_increase.jl")
# No optimisation; no permanent edits to model code.
# Each scenario multiplies the SP DETECTION HAZARD by m, while keeping
# the SN and EP detection hazards at their baseline values exactly.

let
    TB = TBKiribatiJulia
    DE = TB.DoddExtendedStates
    root = normpath(joinpath(@__DIR__, "..", ".."))
    result_dir = joinpath(root, "results", "DoddExtended", "sp_detection_surveillance")
    mkpath(result_dir)

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
    obsmodel = TB.AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=40)
    problem = TB.CalibrationProblem(
        model, setup.demography, observations;
        start_year=1970, end_year=2026,
        age_breaks=setup.age_breaks,
        observation_model=obsmodel,
        solver=TB.Tsit5())

    historical = (
        beta=10.77585425024837,
        Lambda0=0.004806008276461904,
        nu=0.014553334963504224,
        omega=0.0043400219072547595,
        detection_logitK=-0.36666323713625854,
        detection_trend=0.0,
    )
    baseline = merge(
        map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (sigma_obs=0.30, tau_year=0.18),
        historical,
    )
    @assert iszero(baseline.detection_trend) "Exact scaling needs zero detection trend"
    @assert baseline.smear_negative_detection == 1.0
    @assert baseline.extrapulmonary_detection == 1.0

    basep = TB.make_parameters(model, baseline)
    q0 = TB.detection_probability(basep.detection, 2023.0)
    odds0 = q0 / (1 - q0)
    maximum = basep.detection.maximum

    # Multiply the common detection odds by m and reduce the SN/EP
    # relative multipliers by 1/m. Thus SP is scaled by m while
    # SN/EP remain EXACTLY constant at every age and year.
    # Invert p = maximum * logistic(detection_logitK).
    function scaled_theta(m)
        target_odds = m * odds0
        target_p = target_odds / (1 + target_odds)
        target_p < maximum || error("m=$m exceeds detection maximum")
        K = target_p / maximum
        logitK = log(K / (1 - K))
        return merge(baseline, (
            detection_logitK=logitK,
            smear_negative_detection=1/m,
            extrapulmonary_detection=1/m,
        ))
    end

    scenarios = [1.0, 1.5, 2.0, 3.0, 4.0]
    years = collect(2013:2024)
    sites = ("SP", "SN", "EP")
    stride = TB.nstates(TB.state_layout(model))
    breaks = setup.age_breaks

    counter_diff(u0, u1, age, state) = Float64(
        u1[(age-1)*stride+state] - u0[(age-1)*stride+state])

    national_rows = NamedTuple[]
    site_rows = NamedTuple[]
    age_rows = NamedTuple[]
    risk_rows = NamedTuple[]

    for m in scenarios
        theta = scaled_theta(m)
        p = TB.make_parameters(model, theta)
        q = TB.detection_probability(p.detection, 2023.0)
        common_odds = q / (1-q)
        actual_ratio = common_odds / odds0
        @assert isapprox(actual_ratio, m; rtol=1e-10)
        @assert isapprox(actual_ratio*p.relative_detection[2], 1.0; rtol=1e-10)
        @assert isapprox(actual_ratio*p.relative_detection[3], 1.0; rtol=1e-10)
        println("\nSimulating SP detection ×", m, " (SN and EP unchanged)")
        sol = TB.simulate_from_draw(problem.simulation,
            TB.model_draw(problem, theta))
        successful_retcode(sol) || error("Simulation failed for m=$m: $(sol.retcode)")
        out = TB.model_outputs(model, sol;
            variables=(:population, :incidence, :notifications,
                :mortality, :untreated_mortality, :treatment_mortality,
                :latent_prevalence, :recurrent_notifications),
            years=years)

        for (j, year) in enumerate(years)
            total(v) = sum(@view getproperty(out, v)[:, j])
            pop = total(:population)
            push!(national_rows, (
                sp_detection_multiplier=m, year=year, population=pop,
                incidence=total(:incidence),
                incidence_per_100k=1e5*total(:incidence)/pop,
                notifications=total(:notifications),
                untreated_deaths=total(:untreated_mortality),
                treatment_deaths=total(:treatment_mortality),
                tb_deaths=total(:mortality),
                latent_pct=100*total(:latent_prevalence)/pop,
                recurrent_notifications=total(:recurrent_notifications),
            ))

            u0, u1 = sol(Float64(year)), sol(Float64(year+1))
            for k in 1:3
                inc = deaths = cures = notifs = 0.0
                for a in 1:length(p.nu)
                    inc += counter_diff(u0, u1, a, DE.CumIncidenceSP+k-1)
                    deaths += counter_diff(u0, u1, a, DE.CumUntreatedDeathsSP+k-1)
                    cures += counter_diff(u0, u1, a, DE.CumSelfCureSP+k-1)
                    notifs += counter_diff(u0, u1, a, DE.CumNewSP+k-1)
                    notifs += counter_diff(u0, u1, a, DE.CumRecurrentSP+k-1)
                end
                push!(site_rows, (
                    sp_detection_multiplier=m, year=year,
                    presentation=sites[k], incidence=inc,
                    notifications=notifs, untreated_deaths=deaths,
                    self_cures=cures,
                ))
            end

            for (g, label) in enumerate(setup.age_labels)
                ages = (breaks[g]+1):breaks[g+1]
                push!(age_rows, (
                    sp_detection_multiplier=m, year=year,
                    age_group=label,
                    incidence=sum(@view out.incidence[ages, j]),
                    notifications=sum(@view out.notifications[ages, j]),
                    untreated_deaths=sum(@view out.untreated_mortality[ages, j]),
                    tb_deaths=sum(@view out.mortality[ages, j]),
                ))
            end
        end

        for k in 1:3
            a = 26  # age 25, an adult with unit age detection multiplier
            delta = common_odds*p.detection_multiplier[a]*p.relative_detection[k]
            gamma = p.self_cure[a, k]
            mu = p.tb_mortality[a, k]
            z = delta+gamma+mu
            push!(risk_rows, (
                sp_detection_multiplier=m, presentation=sites[k],
                detection_hazard=delta, self_cure_hazard=gamma,
                mortality_hazard=mu,
                p_detection=delta/z, p_self_cure=gamma/z, p_death=mu/z,
            ))
        end
    end

    national_df = DataFrame(national_rows)
    site_df = DataFrame(site_rows)
    age_df = DataFrame(age_rows)
    risks_df = DataFrame(risk_rows)

    # Verify site-specific counter totals agree with national model outputs.
    for r in eachrow(national_df)
        s = site_df[(site_df.year .== r.year) .& (site_df.sp_detection_multiplier .== r.sp_detection_multiplier), :]
        @assert isapprox(sum(s.notifications), r.notifications; rtol=1e-5, atol=1e-3)
        @assert isapprox(sum(s.untreated_deaths), r.untreated_deaths; rtol=1e-5, atol=1e-3)
    end

    CSV.write(joinpath(result_dir, "national_comparison.csv"), national_df)
    CSV.write(joinpath(result_dir, "presentation_comparison.csv"), site_df)
    CSV.write(joinpath(result_dir, "age_comparison.csv"), age_df)
    CSV.write(joinpath(result_dir, "adult_competing_risks.csv"), risks_df)

    println("\n2023 NATIONAL COMPARISON (full simulations)")
    display(select(national_df[national_df.year .== 2023, :],
        :sp_detection_multiplier, :incidence, :notifications,
        :untreated_deaths, :treatment_deaths, :tb_deaths,
        :incidence_per_100k, :latent_pct))
    println("\n2023 PRESENTATION-SPECIFIC COUNTS")
    display(site_df[site_df.year .== 2023, :])
    println("\n2023 ADULT COMPETING-RISK PROBABILITIES")
    display(risks_df)
    println("\n2023 NOTIFICATIONS AND MORTALITY BY AGE")
    display(age_df[age_df.year .== 2023, :])
    println("\nSaved simulated outputs: ", result_dir)

    # ---- Surveillance comparisons (conditional compositions, not calibration) ----
    surveillance_dir = joinpath(root, "data", "processed", "surveillance")
    site_obs = CSV.read(joinpath(surveillance_dir, "tb_notifications_by_site.csv"), DataFrame)
    age_obs = CSV.read(joinpath(surveillance_dir, "tb_notifications_by_age.csv"), DataFrame)
    national_obs = CSV.read(joinpath(surveillance_dir, "tb_notifications_total.csv"), DataFrame)

    # Data classification warning:
    # observed pulmonary_lab_confirmed IS NOT synonymous with smear positive;
    # observed pulmonary_clinically_diagnosed IS NOT synonymous with smear negative.
    # Only total pulmonary versus extrapulmonary is directly comparable.
    println("\nIMPORTANT: 'pulmonary_lab_confirmed' is NOT a smear-positive category.")
    println("The site comparison scores only pulmonary (SP+SN) vs EP.")
    println("SP share vs laboratory-confirmed share is descriptive ONLY, not an exact comparison.")

    national_fit_rows = NamedTuple[]
    site_fit_rows = NamedTuple[]
    age_fit_rows = NamedTuple[]
    year_scores = NamedTuple[]

    # Poisson deviance contribution for descriptive comparison only.
    # This is NOT the model's NB2/random-effect calibration log-likelihood.
    function poisson_deviance(y, mu)
        mu > 0 || return Inf
        return 2 * (y == 0 ? mu : y*log(y/mu) - y + mu)
    end
    function binom_deviance(y, n, p)
        (n >= 0 && 0 <= y <= n && 0 < p < 1) || return Inf
        a = y == 0 ? 0.0 : y*log(y/(n*p))
        b = y == n ? 0.0 : (n-y)*log((n-y)/(n*(1-p)))
        return 2*(a+b)
    end
    # Observed age groups can change by year (notably 2024 has 0-14).
    # Recognise only complete, non-overlapping partitions of model's eight bands.
    band_labels = String.(setup.age_labels)
    function compatible_model_groups(label::AbstractString)
        if label == "0-14"
            return ["0-4", "5-14"]
        elseif label in band_labels
            return [label]
        end
        return String[]
    end

    for m in scenarios, year in years
        nat = national_df[(national_df.year .== year) .& (national_df.sp_detection_multiplier .== m), :]
        @assert nrow(nat) == 1
        pred_total = Float64(nat.notifications[1])
        obs_nat = national_obs[national_obs.year .== year, :]
        if nrow(obs_nat) == 1
            actual = Float64(obs_nat.cases[1])
            push!(national_fit_rows, (
                sp_detection_multiplier=m, year=year,
                observed=actual, predicted=pred_total,
                residual=pred_total-actual,
                poisson_deviance=poisson_deviance(actual, pred_total),
            ))
        end

        site = site_obs[site_obs.year .== year, :]
        sf = site_df[(site_df.year .== year) .& (site_df.sp_detection_multiplier .== m), :]
        if nrow(site) == 1
            pulmonary = Float64(site.pulmonary_lab_confirmed[1] +
                site.pulmonary_clinically_diagnosed[1])
            ep = Float64(site.extrapulmonary[1])
            nsite = pulmonary + ep
            model_sp = sf.notifications[sf.presentation .== "SP"][1]
            model_sn = sf.notifications[sf.presentation .== "SN"][1]
            model_ep = sf.notifications[sf.presentation .== "EP"][1]
            model_pulm = model_sp + model_sn
            model_total = model_pulm + model_ep
            model_ep_fraction = model_ep/model_total
            obs_lab_fraction = Float64(site.pulmonary_lab_confirmed[1])/nsite
            push!(site_fit_rows, (
                sp_detection_multiplier=m, year=year,
                observed_pulmonary=pulmonary, observed_ep=ep,
                observed_total=nsite, observed_ep_fraction=ep/nsite,
                predicted_pulmonary=model_pulm, predicted_ep=model_ep,
                predicted_ep_fraction=model_ep_fraction,
                ep_fraction_error=model_ep_fraction-ep/nsite,
                conditional_binomial_deviance=binom_deviance(ep, nsite, model_ep_fraction),
                observed_lab_confirmed_fraction=obs_lab_fraction,
                predicted_sp_fraction=model_sp/model_total,
                # OBSERVED LAB-CONFIRMED IS NOT EQUIVALENT TO MODEL SP.
                lab_vs_sp_difference_descriptive=model_sp/model_total-obs_lab_fraction,
                observed_site_vs_national_difference=nsite -
                    (nrow(obs_nat) == 1 ? Float64(obs_nat.cases[1]) : NaN),
            ))
        end

        ar = age_obs[age_obs.year .== year, :]
        af = age_df[(age_df.year .== year) .& (age_df.sp_detection_multiplier .== m), :]
        expanded = NamedTuple[]
        seen = String[]
        ok = true
        for o in eachrow(ar)
            group = string(o.age_group)
            component_bands = compatible_model_groups(group)
            if isempty(component_bands) || any(b -> b in seen, component_bands)
                ok = false
                break
            end
            append!(seen, component_bands)
            prediction = sum(af.notifications[in.(af.age_group, Ref(component_bands))])
            push!(expanded, (age_group=group, observed=Float64(o.cases),
                predicted=Float64(prediction)))
        end
        if !ok || sort(seen) != sort(band_labels)
            println("Skipping incomplete/nonstandard age partition year=$year (m=$m): ",
                join(string.(ar.age_group), ", "))
            continue
        end
        obs_age_total = sum(x.observed for x in expanded)
        model_age_total = sum(x.predicted for x in expanded)
        age_deviance = 0.0
        for x in expanded
            expected_conditional = obs_age_total*x.predicted/model_age_total
            d = poisson_deviance(x.observed, expected_conditional)
            age_deviance += d
            push!(age_fit_rows, (
                sp_detection_multiplier=m, year=year,
                age_group=x.age_group,
                observed=x.observed, predicted=x.predicted,
                predicted_conditional=expected_conditional,
                residual=x.predicted-x.observed,
                conditional_deviance_contribution=d,
                observed_age_total=obs_age_total,
            ))
        end
        push!(year_scores, (
            sp_detection_multiplier=m, year=year,
            age_multinomial_deviance=age_deviance,
            observed_age_total=obs_age_total,
            predicted_age_total=model_age_total,
        ))
    end

    national_fit = DataFrame(national_fit_rows)
    site_fit = DataFrame(site_fit_rows)
    age_fit = DataFrame(age_fit_rows)
    age_scores = DataFrame(year_scores)

    summary_rows = NamedTuple[]
    for m in scenarios
        nf = national_fit[national_fit.sp_detection_multiplier .== m, :]
        sf = site_fit[site_fit.sp_detection_multiplier .== m, :]
        as = age_scores[age_scores.sp_detection_multiplier .== m, :]
        af = age_fit[age_fit.sp_detection_multiplier .== m, :]
        v23 = site_fit[(site_fit.sp_detection_multiplier .== m) .& (site_fit.year .== 2023), :]
        nat23 = national_df[(national_df.sp_detection_multiplier .== m) .& (national_df.year .== 2023), :]
        push!(summary_rows, (
            sp_detection_multiplier=m,
            national_years=nrow(nf), site_years=nrow(sf), age_years=nrow(as),
            national_poisson_deviance=sum(nf.poisson_deviance),
            national_mae=mean(abs.(nf.residual)),
            pulmonary_ep_binomial_deviance=sum(sf.conditional_binomial_deviance),
            pulmonary_ep_fraction_mae=mean(abs.(sf.ep_fraction_error)),
            age_multinomial_deviance=sum(as.age_multinomial_deviance),
            age_count_mae=mean(abs.(af.residual)),
            notifications_2023=nat23.notifications[1],
            deaths_2023=nat23.tb_deaths[1],
            observed_ep_fraction_2023=v23.observed_ep_fraction[1],
            predicted_ep_fraction_2023=v23.predicted_ep_fraction[1],
        ))
    end
    scores = DataFrame(summary_rows)
    CSV.write(joinpath(result_dir, "fit_national_2013_2024.csv"), national_fit)
    CSV.write(joinpath(result_dir, "fit_pulmonary_ep_2013_2024.csv"), site_fit)
    CSV.write(joinpath(result_dir, "fit_age_2013_2024.csv"), age_fit)
    CSV.write(joinpath(result_dir, "age_deviance_by_year.csv"), age_scores)
    CSV.write(joinpath(result_dir, "fit_summary.csv"), scores)

    println("\nSCENARIO COMPARISON, 2013–2024")
    display(select(scores,
        :sp_detection_multiplier, :national_poisson_deviance,
        :pulmonary_ep_binomial_deviance, :age_multinomial_deviance,
        :national_mae, :pulmonary_ep_fraction_mae, :age_count_mae))
    println("\n2023 OBSERVED VS PREDICTED EP FRACTION (VALID COMPARISON)")
    display(select(scores, :sp_detection_multiplier,
        :observed_ep_fraction_2023, :predicted_ep_fraction_2023,
        :notifications_2023, :deaths_2023))
    println("\n2023 AGE NOTIFICATION FIT")
    display(select(age_fit[age_fit.year .== 2023, :],
        :sp_detection_multiplier, :age_group, :observed, :predicted))
    println("\nYEAR-BY-YEAR NATIONAL AND SITE COMPARISONS")
    display(national_fit)
    display(select(site_fit, :sp_detection_multiplier, :year,
        :observed_ep_fraction, :predicted_ep_fraction,
        :conditional_binomial_deviance))
    println("\nCaution: the three deviance measures are diagnostics, not a joint")
    println("likelihood, and must not be added or interpreted as calibrated evidence.")
    println("Saved observed-versus-predicted CSVs: ", result_dir)
end
