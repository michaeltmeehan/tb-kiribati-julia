using TBKiribatiJulia
using CSV, DataFrames
using Distributions: Normal, truncated, median
using SciMLBase: successful_retcode

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
    result_dir = joinpath(root, "results", "DoddExtended", "sp_detection_increase")
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
    years = [2013, 2019, 2023, 2024]
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
    println("\nSaved outputs: ", result_dir)
end
