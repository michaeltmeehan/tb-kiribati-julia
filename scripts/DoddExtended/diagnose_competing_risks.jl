using TBKiribatiJulia
using CSV, DataFrames
using Distributions: Normal, truncated, median
using SciMLBase: successful_retcode

# Standalone competing-risk diagnostic for the CURRENT DoddExtended model.
# Uses historical fitted beta, nu, omega and detection parameters; NO fitting.
# Place at scripts/DoddExtended/diagnose_competing_risks.jl and run:
#   include("scripts/DoddExtended/diagnose_competing_risks.jl")
# Outputs: results/DoddExtended/competing_risk_diagnostics/

let
    TB = TBKiribatiJulia
    DE = TB.DoddExtendedStates
    root = normpath(joinpath(@__DIR__, "..", ".."))
    result_dir = joinpath(root, "results", "DoddExtended", "competing_risk_diagnostics")
    mkpath(result_dir)

    include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))
    setup = setup_dodd_analysis(
        start_year=1970, end_year=2026, output_years=2013:2024,
    )
    presentation = TB.load_kiribati_presentation(
        setup.surveillance_dir; years=2007:2012,
    )
    model = TB.DoddExtendedTBModel(
        setup.contact, presentation.presentation; reinfection=true,
    )
    national = filter(
        r -> 1980 <= r.year <= 2025,
        setup.surveillance.national_notifications,
    )
    observations = TB.combine_notification_observations(
        setup.age_observations, national;
        n_age_groups=length(setup.age_labels),
        start_year=1970, end_year=2026,
    )
    observation_model = TB.AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=40,
    )
    problem = TB.CalibrationProblem(
        model, setup.demography, observations;
        start_year=1970, end_year=2026,
        age_breaks=setup.age_breaks,
        observation_model=observation_model,
        solver=TB.Tsit5(),
    )
    historical = (
        beta=10.77585425024837,
        Lambda0=0.004806008276461904,
        nu=0.014553334963504224,
        omega=0.0043400219072547595,
        detection_logitK=-0.36666323713625854,
        detection_trend=0.0,
    )
    theta = merge(
        map(median, problem.prior),
        TB.extended_fixed_parameters(),
        (sigma_obs=0.30, tau_year=0.18),
        historical,
    )
    p = TB.make_parameters(model, theta)

    println("\nCOMPETING-RISK DIAGNOSTICS: historical parameters in current model")
    println("No optimisation. Simulation: 1970–2026. Presentation: SP, SN, EP.")
    println("beta=$(theta.beta), nu=$(theta.nu), omega=$(theta.omega)")
    println("Current baseline hazards: SP gamma=$(p.self_cure[26,1]), mu=$(p.tb_mortality[26,1])")
    println("                         SN gamma=$(p.self_cure[26,2]), mu=$(p.tb_mortality[26,2])")
    println("                         EP gamma=$(p.self_cure[26,3]), mu=$(p.tb_mortality[26,3])")

    sol = TB.simulate_from_draw(problem.simulation, TB.model_draw(problem, theta))
    successful_retcode(sol) || error("Simulation failed: $(sol.retcode)")

    # Cumulative, age- and presentation-specific event counters are indexed
    # directly. Each 2023 annual flow is counter(2024) - counter(2023).
    # This uses the model's actual simulated flows, NOT stationary approximations.
    stride = TB.nstates(TB.state_layout(model))
    breaks = setup.age_breaks
    labels = setup.age_labels
    sites = ("SP", "SN", "EP")
    years = (2013, 2019, 2023, 2024)

    function counter_diff(u0, u1, a, state)
        j = (a - 1) * stride + state
        return max(0.0, Float64(u1[j] - u0[j]))
    end

    # Snapshot risks. These are not automatically equal to yearly counts
    # divided by annual incidence because of lagged active disease, ageing,
    # secular variation, and competing events during ongoing disease.
    function risk_probs(a, k, year; det_scale=1.0, death_scale=1.0, cure_scale=1.0)
        q = TB.detection_probability(p.detection, year)
        delta = (q / (1 - q)) * p.detection_multiplier[a] * p.relative_detection[k] * det_scale
        gamma = p.self_cure[a, k] * cure_scale
        mu = p.tb_mortality[a, k] * death_scale
        z = delta + gamma + mu
        return (
            detection_hazard=delta, self_cure_hazard=gamma, mortality_hazard=mu,
            p_detection=delta/z, p_self_cure=gamma/z, p_death=mu/z,
        )
    end

    rows = NamedTuple[]
    for year in years
        u0, u1 = sol(Float64(year)), sol(Float64(year+1))
        for (g, label) in enumerate(labels)
            for k in 1:3
                onset = 0.0
                deaths = 0.0
                cures = 0.0
                # Use age and presentation-specific onset flows as weights
                weighted_detection = 0.0
                weighted_cure = 0.0
                weighted_death = 0.0
                weighted_delta = 0.0
                weighted_gamma = 0.0
                weighted_mu = 0.0
                for a in (breaks[g]+1):breaks[g+1]
                    inc = counter_diff(u0, u1, a, DE.CumIncidenceSP + k - 1)
                    deaths += counter_diff(u0, u1, a, DE.CumUntreatedDeathsSP + k - 1)
                    cures += counter_diff(u0, u1, a, DE.CumSelfCureSP + k - 1)
                    # Notification counters are not necessarily split by
                    # presentation; actual starts are reported nationally below.
                    prob = risk_probs(a, k, year + 0.5)
                    onset += inc
                    weighted_detection += inc * prob.p_detection
                    weighted_cure += inc * prob.p_self_cure
                    weighted_death += inc * prob.p_death
                    weighted_delta += inc * prob.detection_hazard
                    weighted_gamma += inc * prob.self_cure_hazard
                    weighted_mu += inc * prob.mortality_hazard
                end
                push!(rows, (
                    year=year, age_group=label, presentation=sites[k],
                    incidence=onset, actual_untreated_deaths=deaths,
                    actual_self_cures=cures,
                    detection_hazard=onset>0 ? weighted_delta/onset : NaN,
                    self_cure_hazard=onset>0 ? weighted_gamma/onset : NaN,
                    mortality_hazard=onset>0 ? weighted_mu/onset : NaN,
                    p_detection=onset>0 ? weighted_detection/onset : NaN,
                    p_self_cure=onset>0 ? weighted_cure/onset : NaN,
                    p_death=onset>0 ? weighted_death/onset : NaN,
                    approx_detected_onsets=weighted_detection,
                    approx_self_cured_onsets=weighted_cure,
                    approx_fatal_onsets=weighted_death,
                ))
            end
        end
    end
    detail = DataFrame(rows)
    CSV.write(joinpath(result_dir, "age_presentation_competing_risks.csv"), detail)

    site_summary = combine(groupby(detail, [:year, :presentation]),
        :incidence => sum => :incidence,
        :actual_untreated_deaths => sum => :actual_untreated_deaths,
        :actual_self_cures => sum => :actual_self_cures,
        :approx_detected_onsets => sum => :approx_detected_onsets,
        :approx_fatal_onsets => sum => :approx_fatal_onsets,
    )
    site_summary.p_detection = site_summary.approx_detected_onsets ./ site_summary.incidence
    site_summary.p_death = site_summary.approx_fatal_onsets ./ site_summary.incidence
    CSV.write(joinpath(result_dir, "presentation_death_contributions.csv"), site_summary)

    age_summary = combine(groupby(detail, [:year, :age_group]),
        :incidence => sum => :incidence,
        :actual_untreated_deaths => sum => :actual_untreated_deaths,
        :actual_self_cures => sum => :actual_self_cures,
        :approx_detected_onsets => sum => :approx_detected_onsets,
        :approx_fatal_onsets => sum => :approx_fatal_onsets,
    )
    CSV.write(joinpath(result_dir, "age_death_contributions.csv"), age_summary)

    annual = TB.model_outputs(model, sol;
        variables=(:incidence, :notifications, :mortality,
            :untreated_mortality, :treatment_mortality),
        years=collect(years),
    )
    national_rows = NamedTuple[]
    for (j, year) in enumerate(years)
        tot(v) = sum(@view getproperty(annual, v)[:, j])
        counted = detail[detail.year .== year, :]
        push!(national_rows, (
            year=year, incidence=tot(:incidence), notifications=tot(:notifications),
            untreated_deaths=tot(:untreated_mortality),
            treatment_deaths=tot(:treatment_mortality),
            all_tb_deaths=tot(:mortality),
            self_cure_from_site_counters=sum(counted.actual_self_cures),
            site_counter_untreated_deaths=sum(counted.actual_untreated_deaths),
            site_counter_self_cures=sum(counted.actual_self_cures),
            approx_detected_onsets=sum(counted.approx_detected_onsets),
            approx_fatal_onsets=sum(counted.approx_fatal_onsets),
        ))
    end
    national_df = DataFrame(national_rows)
    CSV.write(joinpath(result_dir, "national_flow_check.csv"), national_df)

    # Guard against inconsistent conventions in cumulative counter extraction.
    # Differences beyond numerical tolerance indicate that results should be
    # checked before interpreting age/site decomposition.
    for r in eachrow(national_df)
        d = abs(r.untreated_deaths-r.site_counter_untreated_deaths)
        if d > 0.01 * max(r.untreated_deaths, 1.0)
            @warn "Untreated-death counters differ from model_outputs: check annual extraction conventions" year=r.year death_difference=d
        end
    end

    # One-at-a-time instantaneous competing-risk sensitivities.
    # Hold simulated incidence by age/presentation FIXED. These are not new
    # epidemic simulations, predictions, or fits.
    scenarios = (
        (name="baseline", det=1.0, death=1.0, cure=1.0),
        (name="detection_x1.5", det=1.5, death=1.0, cure=1.0),
        (name="detection_x2", det=2.0, death=1.0, cure=1.0),
        (name="mortality_x0.75", det=1.0, death=0.75, cure=1.0),
        (name="mortality_x0.5", det=1.0, death=0.5, cure=1.0),
        (name="self_cure_x1.5", det=1.0, death=1.0, cure=1.5),
        (name="self_cure_x2", det=1.0, death=1.0, cure=2.0),
        (name="self_cure_x4", det=1.0, death=1.0, cure=4.0),
    )
    sensitivities = NamedTuple[]
    y = 2023
    u0, u1 = sol(Float64(y)), sol(Float64(y+1))
    for s in scenarios
        dsum = csum = msum = onsets = 0.0
        for a in 1:(length(p.nu)), k in 1:3
            inc = counter_diff(u0, u1, a, DE.CumIncidenceSP + k - 1)
            risk = risk_probs(a, k, y+0.5;
                det_scale=s.det, death_scale=s.death, cure_scale=s.cure)
            dsum += inc*risk.p_detection
            csum += inc*risk.p_self_cure
            msum += inc*risk.p_death
            onsets += inc
        end
        push!(sensitivities, (
            scenario=s.name, detection_multiplier=s.det,
            mortality_multiplier=s.death, self_cure_multiplier=s.cure,
            incident_cases=onsets,
            approx_detected_onsets=dsum,
            approx_self_cured_onsets=csum,
            approx_untreated_fatal_onsets=msum,
            death_to_detection_ratio=msum/dsum,
            p_detection=dsum/onsets, p_death=msum/onsets,
        ))
    end
    sensitivity_df = DataFrame(sensitivities)
    CSV.write(joinpath(result_dir, "competing_risk_sensitivity_2023.csv"), sensitivity_df)

    println("\nNATIONAL ANNUAL FLOWS (model totals versus site-specific counters)")
    display(national_df)
    println("\n2023 MORTALITY BY CLINICAL PRESENTATION")
    display(site_summary[site_summary.year .== 2023, :])
    println("\n2023 UNTREATED DEATHS BY AGE")
    display(age_summary[age_summary.year .== 2023, :])
    println("\n2023 AGE × PRESENTATION COMPETING RISKS")
    display(select(detail[detail.year .== 2023, :],
        :age_group, :presentation, :incidence, :actual_untreated_deaths,
        :detection_hazard, :self_cure_hazard, :mortality_hazard,
        :p_detection, :p_self_cure, :p_death))
    println("\n2023 ONE-AT-A-TIME SENSITIVITIES (incidence held fixed)")
    display(sensitivity_df)
    println("\nSaved CSV files to: ", result_dir)
    println("NOTE: Scenario results are stationary competing-risk approximations, not resimulations.")
end
