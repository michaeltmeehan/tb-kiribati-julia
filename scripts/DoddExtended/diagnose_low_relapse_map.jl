using TBKiribatiJulia
using CSV, DataFrames
using Distributions: Normal, truncated, median
using SciMLBase: successful_retcode

# Standalone diagnostics for the saved notification-only, low-relapse MAP.
# No optimisation is performed.
# Run from repository root:
#   include("scripts/DoddExtended/diagnose_low_relapse_map.jl")

let
    TB = TBKiribatiJulia
    DE = TB.DoddExtendedStates
    start_year, end_year = 1970, 2026
    years = [1980, 2000, 2013, 2019, 2023, 2024]
    fit_dir = joinpath("results", "DoddExtended", "notification_only_low_relapse_map")
    param_path = joinpath(fit_dir, "parameters.csv")
    isfile(param_path) || error("Missing $param_path; run notification_only_low_relapse_map.jl first")

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
    observations = TB.combine_notification_observations(
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
        model, setup.demography, observations;
        start_year=start_year,
        end_year=end_year,
        age_breaks=setup.age_breaks,
        observation_model=observation_model,
        solver=TB.Tsit5(),
    )

    saved = CSV.read(param_path, DataFrame)
    @assert all(k -> k in propertynames(saved), (:parameter, :value))
    saved_parameters = NamedTuple{Tuple(Symbol.(saved.parameter))}(Tuple(saved.value))
    theta = merge(map(median, problem.prior), saved_parameters)
    @assert isapprox(theta.omega, 0.001; atol=1e-12) "Unexpected relapse hazard in saved fit"
    p = TB.make_parameters(model, theta)

    println("Running one simulation for saved MAP parameters...")
    sol = TB.simulate_from_draw(problem.simulation, TB.model_draw(problem, theta))
    successful_retcode(sol) || error("Simulation failed: $(sol.retcode)")

    variables = (
        :population, :incidence, :notifications, :recurrent_notifications,
        :primary_infections, :reinfections,
        :fast_progression, :slow_progression, :relapse,
        :untreated_active_prevalence, :latent_prevalence,
        :early_latent_prevalence, :late_latent_prevalence,
    )
    # Single-year-age output matrices (96 ages × years).
    out = TB.model_outputs(model, sol; variables=variables, years=years)
    ex = TB.OutputExtraction(sol, model; years=years)
    stocks = (
        susceptible=TB.stock_output(ex, (DE.S,)),
        early_never_treated=TB.stock_output(ex, (DE.E0,)),
        early_previously_treated=TB.stock_output(ex, (DE.E1,)),
        late_never_treated=TB.stock_output(ex, (DE.L0,)),
        late_previously_treated=TB.stock_output(ex, (DE.L1,)),
        recovered=TB.stock_output(ex, (DE.R,)),
        on_treatment=TB.stock_output(ex, (DE.T,)),
    )

    breaks = problem.age_breaks
    age_rows = NamedTuple[]
    national_rows = NamedTuple[]
    for (j, year) in enumerate(years)
        # Presentation-specific probability of detection before self-cure/death,
        # conditional on active disease at a given age and year.
        ascertainment = TB.presentation_ascertainment(p, year)
        for g in 1:(length(breaks)-1)
            ages = (breaks[g]+1):breaks[g+1]
            lo, hi = breaks[g], breaks[g+1] - 1
            label = g == length(breaks)-1 ? "$(lo)+" : "$(lo)-$(hi)"
            total(v) = sum(@view getproperty(out, v)[ages, j])
            stock(v) = sum(@view getproperty(stocks, v)[ages, j])
            pop = total(:population)
            inc = total(:incidence)
            notif = total(:notifications)
            recurrent = total(:recurrent_notifications)
            primary = total(:primary_infections)
            reinfections = total(:reinfections)
            fast = total(:fast_progression)
            slow = total(:slow_progression)
            relapse = total(:relapse)
            susceptible = stock(:susceptible)
            early = stock(:early_never_treated) + stock(:early_previously_treated)
            late = stock(:late_never_treated) + stock(:late_previously_treated)
            recovered = stock(:recovered)
            active = total(:untreated_active_prevalence)

            # Disease-onset-weighted probability of diagnosis before spontaneous
            # resolution, assuming constant hazards and no ageing during disease.
            # This is NOT the model's annual notifications / incidence ratio.
            detected_onsets = 0.0
            for a in ages
                af = sum(@view(p.presentation_fast[a, :]) .* ascertainment[a, :])
                as = sum(@view(p.presentation_slow[a, :]) .* ascertainment[a, :])
                ar = sum(@view(p.presentation_relapse[a, :]) .* ascertainment[a, :])
                detected_onsets += out.fast_progression[a, j] * af +
                    out.slow_progression[a, j] * as +
                    out.relapse[a, j] * ar
            end

            push!(age_rows, (
                year=year, age_group=label, population=pop,
                susceptible=susceptible, susceptible_pct=100*susceptible/pop,
                early_latent=early, late_latent=late,
                latent_pct=100*(early+late)/pop,
                recovered=recovered, recovered_pct=100*recovered/pop,
                on_treatment=stock(:on_treatment),
                untreated_active=active, untreated_active_pct=100*active/pop,
                primary_infections=primary, reinfections=reinfections,
                reinfections_per_person_year=reinfections/pop,
                fast_progression=fast, slow_progression=slow, relapse=relapse,
                incidence=inc, fast_fraction=inc > 0 ? fast/inc : NaN,
                notifications=notif, recurrent_notifications=recurrent,
                implied_onset_ascertainment=inc > 0 ? detected_onsets/inc : NaN,
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
            year=year,
            population=pop,
            susceptible_pct=100*sum(rows.susceptible)/pop,
            latent_pct=100*sum(rows.early_latent .+ rows.late_latent)/pop,
            recovered_pct=100*sum(rows.recovered)/pop,
            active_pct=100*sum(rows.untreated_active)/pop,
            primary_infections=sum(rows.primary_infections),
            reinfections=sum(rows.reinfections),
            incidence=inc,
            fast_fraction=sum(rows.fast_progression)/inc,
            notifications=notif,
            recurrent_fraction=sum(rows.recurrent_notifications)/notif,
            implied_onset_ascertainment=sum(rows.implied_detected_onsets)/inc,
        ))
    end
    national_df = DataFrame(national_rows)
    CSV.write(joinpath(fit_dir, "age_mechanisms.csv"), age_df)
    CSV.write(joinpath(fit_dir, "national_mechanisms.csv"), national_df)

    println("\nNATIONAL MECHANISMS")
    display(national_df)
    println("\n2023 AGE-SPECIFIC STOCKS (%)")
    display(select(age_df[age_df.year .== 2023, :],
        :age_group, :susceptible_pct, :latent_pct, :recovered_pct,
        :untreated_active_pct))
    println("\n2023 AGE-SPECIFIC INFECTION AND PROGRESSION")
    display(select(age_df[age_df.year .== 2023, :],
        :age_group, :primary_infections, :reinfections,
        :fast_progression, :slow_progression, :relapse,
        :incidence, :fast_fraction))
    println("\n2023 DETECTION AND NOTIFICATIONS")
    display(select(age_df[age_df.year .== 2023, :],
        :age_group, :incidence, :notifications,
        :implied_onset_ascertainment))
    println("\nFiles saved to: $fit_dir")
end
