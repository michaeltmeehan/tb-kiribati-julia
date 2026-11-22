
using TBKiribatiJulia
using CSV, DataFrames, Statistics

function diagnose_age_dynamics(fit; year=2024)
    TB = TBKiribatiJulia
    problem = fit.problem
    theta = fit.parameters

    variables = (
        :population,
        :early_latent_prevalence,
        :late_latent_prevalence,
        :latent_prevalence,
        :untreated_active_prevalence,
        :incidence,
        :fast_progression,
        :slow_progression,
        :relapse,
        :notifications,
        :untreated_mortality,
    )

    out = predict(
        problem, theta;
        variables=variables,
    )

    j = findfirst(==(year), out.years)
    isnothing(j) && error("Missing year $year")

    breaks = problem.age_breaks
    p = TB.make_parameters(problem.model, theta)

    # Detection odds at the midpoint of the year.
    q = TB.detection_probability(
        p.detection, year + 0.5
    )
    odds = q / (1 - q)

    rows = NamedTuple[]

    for g in 1:(length(breaks)-1)
        a0 = breaks[g]
        a1 = breaks[g+1] - 1
        ages = (a0+1):(a1+1)

        label = g == length(breaks)-1 ?
                "$(a0)+" : "$(a0)-$(a1)"

        value(v) = getproperty(out, v)[g, j]

        pop = value(:population)
        early = value(:early_latent_prevalence)
        late = value(:late_latent_prevalence)
        latent = value(:latent_prevalence)

        fast = value(:fast_progression)
        slow = value(:slow_progression)
        relapse = value(:relapse)
        incidence = value(:incidence)
        notifications = value(:notifications)

        # Presentation-weighted competing-risk detection
        # probability, averaged uniformly over single ages.
        #
        # This is a hazard-based approximation, NOT an exact
        # cohort probability or incidence-weighted estimate.
        detection_risk = mean(
            sum(
                p.presentation[a, k] * begin
                    delta = p.detection_multiplier[a] *
                        odds * p.relative_detection[k]

                    delta / (
                        delta +
                            p.self_cure[a, k] +
                            p.tb_mortality[a, k]
                    )
                end
                for k in 1:3
            )
            for a in ages
        )

        push!(rows, (
            year=year,
            age_group=label,
            population=pop,
            early_latent=early,
            late_latent=late,
            latent=latent,
            latent_fraction=latent / pop,
            late_latent_fraction=late / pop,
            active_prevalence=
            value(:untreated_active_prevalence),
            fast_progression=fast,
            slow_progression=slow,
            relapse=relapse,
            incidence=incidence,
            incidence_per_100k=
            1e5 * incidence / pop,
            slow_progression_fraction=
            slow / incidence,
            notifications=notifications,
            notification_incidence_ratio=
            notifications / incidence,
            competing_risk_detection=
            detection_risk,
            untreated_deaths=
            value(:untreated_mortality),
        ))
    end

    age_dynamics = DataFrame(rows)

    # Verify stock and incidence identities.
    @assert all(isapprox.(
        age_dynamics.latent,
        age_dynamics.early_latent .+
            age_dynamics.late_latent;
        rtol=1e-8,
    ))

    @assert all(isapprox.(
        age_dynamics.incidence,
        age_dynamics.fast_progression .+
            age_dynamics.slow_progression .+
            age_dynamics.relapse;
        rtol=1e-8,
    ))

    # Compare against actual surveillance age categories.
    # This handles the combined 0-14 group in 2024.
    observed = DataFrame(
        TB.notification_fit_rows(problem, out)
    )

    observed = filter(
        r -> r.year == year,
        observed,
    )

    observed.ratio =
        observed.observed ./ observed.expected

    dir = normpath(joinpath(
        @__DIR__, "..", "..",
        "results", "DoddExtended",
        "current_calibration",
    ))

    mkpath(dir)

    CSV.write(
        joinpath(dir, "age_dynamics_$(year).csv"),
        age_dynamics,
    )

    CSV.write(
        joinpath(dir, "age_fit_$(year).csv"),
        observed,
    )

    println("\nAGE-SPECIFIC LATENT INFECTION")
    display(select(
        age_dynamics,
        :age_group,
        :population,
        :early_latent,
        :late_latent,
        :latent_fraction,
    ))

    println("\nAGE-SPECIFIC PROGRESSION")
    display(select(
        age_dynamics,
        :age_group,
        :fast_progression,
        :slow_progression,
        :relapse,
        :incidence,
        :incidence_per_100k,
        :slow_progression_fraction,
    ))

    println("\nAGE-SPECIFIC DETECTION")
    display(select(
        age_dynamics,
        :age_group,
        :notifications,
        :incidence,
        :notification_incidence_ratio,
        :competing_risk_detection,
    ))

    println("\nOBSERVED VERSUS PREDICTED NOTIFICATIONS")
    display(observed)

    return (
        age_dynamics=age_dynamics,
        observed=observed,
    )
end
