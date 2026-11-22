
using TBKiribatiJulia
using CSV, DataFrames, Statistics

function diagnose_presentation(fit)
    problem = fit.problem

    variables = (
        :incidence,
        :notifications,
        :untreated_mortality,
        :self_cure,
        :incidence_sp,
        :incidence_sn,
        :incidence_ep,
        :untreated_mortality_sp,
        :untreated_mortality_sn,
        :untreated_mortality_ep,
        :self_cure_sp,
        :self_cure_sn,
        :self_cure_ep,
        :smear_positive_notifications,
        :smear_negative_notifications,
        :extrapulmonary_notifications,
        :fast_progression,
        :slow_progression,
        :relapse,
    )

    out = predict(problem, fit.parameters; variables=variables)

    breaks = problem.age_breaks
    labels = [
        "$(breaks[g])-$(breaks[g+1]-1)"
        for g in 1:(length(breaks)-1)
    ]
    labels[end] = "$(breaks[end-1])+"

    presentations = (
        ("SP", :incidence_sp,
            :smear_positive_notifications,
            :untreated_mortality_sp, :self_cure_sp),
        ("SN", :incidence_sn,
            :smear_negative_notifications,
            :untreated_mortality_sn, :self_cure_sn),
        ("EP", :incidence_ep,
            :extrapulmonary_notifications,
            :untreated_mortality_ep, :self_cure_ep),
    )

    rows = NamedTuple[]

    for (j, year) in enumerate(out.years)
        for g in eachindex(labels)
            for (site, inc, notif, death, cure) in presentations
                push!(rows, (
                    year=year,
                    age_group=labels[g],
                    presentation=site,
                    incidence=out[inc][g, j],
                    notifications=out[notif][g, j],
                    untreated_deaths=out[death][g, j],
                    self_cure=out[cure][g, j],
                ))
            end
        end
    end

    detail = DataFrame(rows)

    # Verify annual conservation of the recorded event totals.
    checks = NamedTuple[]

    for (j, year) in enumerate(out.years)
        sumsite(prefix) = sum(
            sum(out[Symbol(prefix, suffix)][:, j])
            for suffix in ("sp", "sn", "ep")
        )

        incidence = sum(out.incidence[:, j])
        notifications = sum(out.notifications[:, j])
        deaths = sum(out.untreated_mortality[:, j])
        cures = sum(out.self_cure[:, j])

        inc_error = sumsite("incidence_") - incidence
        death_error =
            sumsite("untreated_mortality_") - deaths
        cure_error = sumsite("self_cure_") - cures

        notification_error =
            sum(out.smear_positive_notifications[:, j]) +
                sum(out.smear_negative_notifications[:, j]) +
                sum(out.extrapulmonary_notifications[:, j]) -
                notifications

        @assert isapprox(inc_error, 0; atol=1e-5)
        @assert isapprox(death_error, 0; atol=1e-5)
        @assert isapprox(cure_error, 0; atol=1e-5)
        @assert isapprox(notification_error, 0; atol=1e-5)

        push!(checks, (
            year=year,
            incidence=incidence,
            notifications=notifications,
            untreated_deaths=deaths,
            self_cure=cures,
            fast_progression=sum(out.fast_progression[:, j]),
            slow_progression=sum(out.slow_progression[:, j]),
            relapse=sum(out.relapse[:, j]),
        ))
    end

    annual = DataFrame(checks)

    # Match exactly the reported notification age categories.
    age_fit = DataFrame(
        TBKiribatiJulia.notification_fit_rows(problem, out)
    )

    age_fit.residual =
        age_fit.observed .- age_fit.expected

    age_fit.pearson_raw = [
        r.expected > 0 ?
        (r.observed - r.expected) / sqrt(r.expected) :
        missing
        for r in eachrow(age_fit)
    ]

    # Summarise 2024 by presentation.
    summary2024 = combine(
        groupby(
            filter(r -> r.year == 2024, detail),
            :presentation,
        ),
        :incidence => sum => :incidence,
        :notifications => sum => :notifications,
        :untreated_deaths => sum => :untreated_deaths,
        :self_cure => sum => :self_cure,
    )

    summary2024.notification_incidence_ratio =
        summary2024.notifications ./ summary2024.incidence

    summary2024.death_incidence_ratio =
        summary2024.untreated_deaths ./ summary2024.incidence

    dir = normpath(joinpath(
        @__DIR__, "..", "..",
        "results", "DoddExtended",
        "current_calibration",
    ))
    mkpath(dir)

    CSV.write(joinpath(dir, "presentation_by_age_year.csv"), detail)
    CSV.write(joinpath(dir, "presentation_annual_totals.csv"), annual)
    CSV.write(joinpath(dir, "age_notification_fit.csv"), age_fit)
    CSV.write(joinpath(dir, "presentation_2024.csv"), summary2024)

    println("\n2024 PRESENTATION BREAKDOWN")
    show(stdout, MIME"text/plain"(), summary2024;
        allrows=true, allcols=true)
    println()

    println("\n2024 AGE-SPECIFIC NOTIFICATION FIT")
    show(
        stdout, MIME"text/plain"(),
        filter(r -> r.year == 2024, age_fit);
        allrows=true, allcols=true,
    )
    println()

    return (
        detail=detail,
        annual=annual,
        age_fit=age_fit,
        summary2024=summary2024,
    )
end
