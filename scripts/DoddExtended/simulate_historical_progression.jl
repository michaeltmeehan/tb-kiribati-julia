using TBKiribatiJulia
using CSV, DataFrames
using Distributions: Normal, truncated, median
using SciMLBase: successful_retcode

# One controlled simulation of the CURRENT DoddExtended model using the
# historically fitted transmission, slow-progression, relapse and detection
# parameters (22 November 2026). No optimisation.
#
# Place in scripts/DoddExtended/ and run from the repository root:
#   include("scripts/DoddExtended/simulate_historical_progression.jl")
#
# Outputs: results/DoddExtended/historical_progression_test/
#
# IMPORTANT: This is NOT a reproduction of the 22-November historical model.
# Only selected historical parameter values are transplanted. All other
# progression, mortality, detection and initialisation definitions remain
# those of the CURRENT code.

let
    TB = TBKiribatiJulia
    DE = TB.DoddExtendedStates
    root = normpath(joinpath(@__DIR__, "..", ".."))
    result_dir = joinpath(root, "results", "DoddExtended", "historical_progression_test")
    mkpath(result_dir)

    start_year, end_year = 1970, 2026
    years = [1980, 2000, 2013, 2019, 2023, 2024, 2025]

    include(joinpath(@__DIR__, "..", "common", "kiribati_setup.jl"))
    setup = setup_dodd_analysis(
        start_year=start_year,
        end_year=end_year,
        output_years=2013:2024,
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
    notifications = TB.combine_notification_observations(
        setup.age_observations, national;
        n_age_groups=length(setup.age_labels),
        start_year=start_year, end_year=end_year,
    )
    observation_model = TB.AnnualRandomEffectNB2Observation(
        truncated(Normal(0.0, 0.3), 0.0, Inf),
        truncated(Normal(0.0, 0.3), 0.0, Inf);
        quadrature_order=40,
    )
    problem = TB.CalibrationProblem(
        model, setup.demography, notifications;
        start_year=start_year, end_year=end_year,
        age_breaks=setup.age_breaks,
        observation_model=observation_model,
        solver=TB.Tsit5(),
    )

    # Historical 22-November 'current_calibration/parameters.csv' fitted values.
    # Remaining biological parameters stay at their CURRENT prior medians.
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

    println("\nHISTORICAL-PROGRESSION TEST — CURRENT DoddExtended MODEL")
    println("No optimisation; reinfection enabled; start/end = $start_year/$end_year")
    println("Historical fitted parameters:")
    for name in keys(historical)
        println("  ", name, " = ", getproperty(theta, name))
    end
    println("Current natural-history assumptions:")
    println("  early_duration = ", theta.early_duration)
    println("  fast_risk_15_plus = ", theta.fast_risk_15_plus)
    println("  adult epsilon = ", p.epsilon[26], " /year")
    println("  adult kappa = ", p.kappa[26], " /year")
    println("  adult nu = ", p.nu[26], " /year")
    println("  detection reference year = ", p.detection.reference_year)

    println("\nRunning simulation...")
    sol = TB.simulate_from_draw(problem.simulation, TB.model_draw(problem, theta))
    successful_retcode(sol) || error("Simulation failed: $(sol.retcode)")

    vars = (
        :population, :incidence, :notifications, :mortality,
        :recurrent_notifications, :primary_infections, :reinfections,
        :fast_progression, :slow_progression, :relapse,
        :untreated_active_prevalence, :latent_prevalence,
        :untreated_mortality, :treatment_mortality,
    )
    out = TB.model_outputs(model, sol; variables=vars, years=years)
    ex = TB.OutputExtraction(sol, model; years=years)
    stocks = (
        susceptible=TB.stock_output(ex, (DE.S,)),
        early_latent=TB.stock_output(ex, (DE.E0, DE.E1)),
        late_latent=TB.stock_output(ex, (DE.L0, DE.L1)),
        recovered=TB.stock_output(ex, (DE.R,)),
        on_treatment=TB.stock_output(ex, (DE.T,)),
    )

    age_rows = NamedTuple[]
    national_rows = NamedTuple[]
    breaks = setup.age_breaks
    for (j, year) in enumerate(years)
        ascertainment = TB.presentation_ascertainment(p, year)
        for g in 1:(length(breaks)-1)
            ages = (breaks[g]+1):breaks[g+1]
            label = setup.age_labels[g]
            total(v) = sum(@view getproperty(out, v)[ages, j])
            stock(v) = sum(@view getproperty(stocks, v)[ages, j])
            pop = total(:population)
            inc = total(:incidence)
            n = total(:notifications)
            fast = total(:fast_progression)
            slow = total(:slow_progression)
            relapse = total(:relapse)

            # Competing-risks ascertainment: probability that a newly diseased
            # individual is diagnosed before self-cure/death (snapshot approx).
            detected_onsets = 0.0
            for a in ages
                qfast = sum(@view(p.presentation_fast[a, :]) .* ascertainment[a, :])
                qslow = sum(@view(p.presentation_slow[a, :]) .* ascertainment[a, :])
                qrelapse = sum(@view(p.presentation_relapse[a, :]) .* ascertainment[a, :])
                detected_onsets += out.fast_progression[a, j]*qfast +
                    out.slow_progression[a, j]*qslow +
                    out.relapse[a, j]*qrelapse
            end

            susceptible = stock(:susceptible)
            early = stock(:early_latent)
            late = stock(:late_latent)
            recovered = stock(:recovered)
            active = total(:untreated_active_prevalence)
            push!(age_rows, (
                year=year, age_group=label, population=pop,
                susceptible=susceptible,
                susceptible_pct=100*susceptible/pop,
                early_latent=early, late_latent=late,
                latent_pct=100*(early+late)/pop,
                recovered=recovered, recovered_pct=100*recovered/pop,
                on_treatment=stock(:on_treatment),
                untreated_active=active, untreated_active_pct=100*active/pop,
                primary_infections=total(:primary_infections),
                reinfections=total(:reinfections),
                fast_progression=fast, slow_progression=slow, relapse=relapse,
                incidence=inc, fast_fraction=inc>0 ? fast/inc : NaN,
                notifications=n,
                recurrent_notifications=total(:recurrent_notifications),
                mortality=total(:mortality),
                untreated_mortality=total(:untreated_mortality),
                treatment_mortality=total(:treatment_mortality),
                implied_onset_ascertainment=inc>0 ? detected_onsets/inc : NaN,
                implied_detected_onsets=detected_onsets,
            ))
        end
    end
    age_df = DataFrame(age_rows)
    for year in years
        rows = age_df[age_df.year .== year, :]
        pop = sum(rows.population)
        inc = sum(rows.incidence)
        notif = sum(rows.notifications)
        push!(national_rows, (
            year=year, population=pop,
            incidence=inc, incidence_per_100k=1e5*inc/pop,
            notifications=notif,
            mortality=sum(rows.mortality),
            untreated_mortality=sum(rows.untreated_mortality),
            treatment_mortality=sum(rows.treatment_mortality),
            susceptible_pct=100*sum(rows.susceptible)/pop,
            latent_pct=100*sum(rows.early_latent .+ rows.late_latent)/pop,
            recovered_pct=100*sum(rows.recovered)/pop,
            active_pct=100*sum(rows.untreated_active)/pop,
            primary_infections=sum(rows.primary_infections),
            reinfections=sum(rows.reinfections),
            fast_progression=sum(rows.fast_progression),
            slow_progression=sum(rows.slow_progression),
            relapse=sum(rows.relapse),
            fast_fraction=inc>0 ? sum(rows.fast_progression)/inc : NaN,
            recurrent_fraction=notif>0 ? sum(rows.recurrent_notifications)/notif : NaN,
            implied_onset_ascertainment=inc>0 ? sum(rows.implied_detected_onsets)/inc : NaN,
        ))
    end
    national_df = DataFrame(national_rows)

    detection_df = DataFrame(
        year=years,
        model_detection_probability=[TB.detection_probability(p.detection, y) for y in years],
    )
    parameters_df = DataFrame(
        parameter=string.(collect(keys(theta))),
        value=Float64.(collect(values(theta))),
        historical_override=[k in keys(historical) for k in keys(theta)],
    )

    CSV.write(joinpath(result_dir, "parameters.csv"), parameters_df)
    CSV.write(joinpath(result_dir, "national_mechanisms.csv"), national_df)
    CSV.write(joinpath(result_dir, "age_mechanisms.csv"), age_df)
    CSV.write(joinpath(result_dir, "detection_diagnostics.csv"), detection_df)

    # The observation data are from the same national/age datasets as recent fits.
    # Use the established reporting helper for year-age observed/predicted rows.
    observed_years = sort(unique(obs.year for obs in problem.observations))

    fit_out = TB.model_outputs(
        model, sol;
        variables=(:notifications,),
        years=observed_years,
        age_breaks=problem.age_breaks,
    )

    age_fit = DataFrame(TB.notification_fit_rows(problem, fit_out))
    CSV.write(joinpath(result_dir, "notification_fit.csv"), age_fit)

    println("\nNATIONAL EPIDEMIOLOGICAL SUMMARY")
    display(select(national_df, :year, :population, :incidence,
        :incidence_per_100k, :notifications, :mortality,
        :susceptible_pct, :latent_pct, :recovered_pct))
    println("\n2023 AGE-SPECIFIC NOTIFICATION FIT (observed vs predicted)")
    display(filter(r -> r.year == 2023, age_fit))
    println("\n2023 AGE-SPECIFIC STOCKS (%)")
    display(select(filter(r -> r.year == 2023, age_df),
        :age_group, :susceptible_pct, :latent_pct,
        :recovered_pct, :untreated_active_pct))
    println("\n2023 AGE-SPECIFIC INFECTIONS / PROGRESSION / DEATHS")
    display(select(filter(r -> r.year == 2023, age_df),
        :age_group, :primary_infections, :reinfections,
        :fast_progression, :slow_progression, :relapse,
        :incidence, :mortality))
    println("\n2023 ASCERTAINMENT AND NOTIFICATIONS")
    display(select(filter(r -> r.year == 2023, age_df),
        :age_group, :incidence, :notifications,
        :implied_onset_ascertainment))
    println("\nDETECTION TRAJECTORY (model probability; not case ascertainment)")
    display(detection_df)
    println("\nOutput directory: ", result_dir)
end
