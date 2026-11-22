
using TBKiribatiJulia
using CSV, DataFrames, CairoMakie
using Random, Statistics
using Distributions: NegativeBinomial

function diagnose_dodd_calibration(fit; nsim=3000)
    problem = fit.problem
    theta = fit.parameters
    rng = MersenneTwister(2026)

    root = normpath(joinpath(@__DIR__, "..", ".."))
    dir = joinpath(root, "results", "DoddExtended", "initial_calibration")
    mkpath(dir)

    # One model simulation, covering the full calibration period.
    years = collect(minimum(o.year for o in problem.observations):
    (problem.end_year-1))

    out = predict(
        problem, theta;
        variables=(
            :notifications,
            :recurrent_notifications,
            :pulmonary_notifications,
            :extrapulmonary_notifications,
            :incidence,
            :mortality,
        ),
        years=years,
    )

    total(variable) = vec(sum(getproperty(out, variable); dims=1))

    model_df = DataFrame(
        year=out.years,
        notifications=total(:notifications),
        recurrent=total(:recurrent_notifications),
        pulmonary=total(:pulmonary_notifications),
        extrapulmonary=total(:extrapulmonary_notifications),
        incidence=total(:incidence),
        mortality=total(:mortality),
    )

    CSV.write(joinpath(dir, "model_annual_outputs.csv"), model_df)

    # --------------------------------------------------
    # 1. Notifications and predictive intervals
    # --------------------------------------------------

    rows = notification_fit_rows(problem, out)
    sigma = theta.sigma_obs
    tau = theta.tau_year

    notif_rows = NamedTuple[]

    for year in sort(unique(r.year for r in rows))
        rr = filter(r -> r.year == year, rows)

        mu = Float64[r.expected for r in rr]
        observed = sum(r.observed for r in rr)
        expected = sum(mu)

        # Shared annual lognormal effect plus conditional NB2.
        simulations = Vector{Int}(undef, nsim)

        for s in eachindex(simulations)
            multiplier = exp(tau * randn(rng) - tau^2 / 2)
            n = 0

            for m in mu
                conditional_mean = m * multiplier
                if conditional_mean > 0
                    phi = 1 / sigma^2
                    p = phi / (phi + conditional_mean)
                    n += rand(rng, NegativeBinomial(phi, p))
                end
            end

            simulations[s] = n
        end

        # Exact marginal variance for the sum of reported groups.
        variance = expected +
            sigma^2 * exp(tau^2) * sum(abs2, mu) +
            (exp(tau^2) - 1) * expected^2

        push!(notif_rows, (
            year=year,
            observed=observed,
            expected=expected,
            lower=quantile(simulations, 0.025),
            upper=quantile(simulations, 0.975),
            residual=observed - expected,
            pearson=(observed - expected) / sqrt(variance),
        ))
    end

    notif = DataFrame(notif_rows)
    CSV.write(joinpath(dir, "notification_diagnostics.csv"), notif)

    fig = Figure(size=(1100, 750))

    ax = Axis(
        fig[1, 1],
        title="Annual TB notifications",
        ylabel="Notifications",
    )

    band!(
        ax, notif.year, notif.lower, notif.upper,
        color=(:steelblue, 0.2),
        label="95% predictive interval",
    )

    lines!(
        ax, notif.year, notif.expected,
        color=:steelblue, linewidth=2,
        label="Marginal expectation",
    )

    scatter!(
        ax, notif.year, notif.observed,
        color=:black, markersize=7,
        label="Observed",
    )

    axislegend(ax, position=:lt)

    ax2 = Axis(
        fig[2, 1],
        xlabel="Year",
        ylabel="Pearson residual",
        title="Marginal standardised residuals",
    )

    hlines!(ax2, [0], color=:gray, linestyle=:dash)
    lines!(ax2, notif.year, notif.pearson, color=:black)
    scatter!(ax2, notif.year, notif.pearson, color=:black)

    save(joinpath(dir, "01_notifications.png"), fig)

    # Residual autocorrelation by lag.
    x = notif.pearson
    acf = DataFrame(lag=Int[], correlation=Float64[])

    for lag in 1:min(10, length(x)-2)
        push!(acf, (
            lag,
            cor(x[1:(end-lag)], x[(1+lag):end]),
        ))
    end

    CSV.write(joinpath(dir, "notification_acf.csv"), acf)

    # --------------------------------------------------
    # 2. Recurrence fraction
    # --------------------------------------------------

    rec_rows = NamedTuple[]

    for r in problem.recurrence.rows
        j = findfirst(==(r.year), out.years)
        predicted_total = sum(out.notifications[:, j])
        predicted_rec = sum(out.recurrent_notifications[:, j])

        push!(rec_rows, (
            year=r.year,
            observed=r.recurrent / r.total,
            predicted=predicted_rec / predicted_total,
            observed_cases=r.recurrent,
            observed_total=r.total,
        ))
    end

    rec = DataFrame(rec_rows)
    CSV.write(joinpath(dir, "recurrence_diagnostics.csv"), rec)

    fig = Figure(size=(1000, 500))
    ax = Axis(
        fig[1, 1],
        xlabel="Year",
        ylabel="Recurrent fraction",
        title="Recurrent TB notifications",
    )

    lines!(ax, rec.year, rec.predicted,
        color=:steelblue, linewidth=2, label="Model")
    scatter!(ax, rec.year, rec.observed,
        color=:black, label="Observed")

    axislegend(ax)
    save(joinpath(dir, "02_recurrence.png"), fig)

    # --------------------------------------------------
    # 3. Pulmonary fraction
    # --------------------------------------------------

    site_rows = NamedTuple[]

    for r in problem.site.rows
        j = findfirst(==(r.year), out.years)
        predicted_pul = sum(out.pulmonary_notifications[:, j])
        predicted_ep = sum(out.extrapulmonary_notifications[:, j])

        push!(site_rows, (
            year=r.year,
            observed=r.pulmonary / r.total,
            predicted=predicted_pul / (predicted_pul + predicted_ep),
            observed_pulmonary=r.pulmonary,
            observed_total=r.total,
        ))
    end

    site = DataFrame(site_rows)
    CSV.write(joinpath(dir, "site_diagnostics.csv"), site)

    fig = Figure(size=(1000, 500))
    ax = Axis(
        fig[1, 1],
        xlabel="Year",
        ylabel="Pulmonary fraction",
        title="Pulmonary TB notifications",
    )

    lines!(ax, site.year, site.predicted,
        color=:steelblue, linewidth=2, label="Model")
    scatter!(ax, site.year, site.observed,
        color=:black, label="Observed")

    axislegend(ax)
    save(joinpath(dir, "03_pulmonary.png"), fig)

    # --------------------------------------------------
    # 4. Model burden versus WHO estimates
    # --------------------------------------------------

    who_file = joinpath(
        root, "data", "processed", "surveillance",
        "estimates", "tb_burden_national.csv",
    )

    who = CSV.read(who_file, DataFrame)
    sort!(who, :year)

    fig = Figure(size=(1100, 750))

    ax = Axis(
        fig[1, 1],
        ylabel="Annual incident cases",
        title="TB incidence: model versus WHO estimates",
    )

    band!(
        ax, who.year, who.e_inc_num_lo, who.e_inc_num_hi,
        color=(:gray, 0.2),
    )

    lines!(
        ax, who.year, who.e_inc_num,
        color=:gray, linestyle=:dash, label="WHO estimate",
    )

    lines!(
        ax, model_df.year, model_df.incidence,
        color=:steelblue, linewidth=2, label="Model",
    )

    axislegend(ax)

    ax2 = Axis(
        fig[2, 1],
        xlabel="Year",
        ylabel="Annual TB deaths",
        title="TB mortality: model versus WHO estimates",
    )

    band!(
        ax2, who.year, who.e_mort_num_lo, who.e_mort_num_hi,
        color=(:gray, 0.2),
    )

    lines!(
        ax2, who.year, who.e_mort_num,
        color=:gray, linestyle=:dash, label="WHO estimate",
    )

    lines!(
        ax2, model_df.year, model_df.mortality,
        color=:steelblue, linewidth=2, label="Model",
    )

    axislegend(ax2)

    save(joinpath(dir, "04_burden_vs_who.png"), fig)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    println("\nCalibration diagnostics")
    println("-----------------------")
    println("Converged: ", fit.converged)
    println("Log likelihood: ", fit.log_likelihood)
    println("Log posterior: ", fit.log_posterior)

    println("\nNotification coverage:")
    println(
        "Within 95% predictive interval: ",
        round(mean(
            (notif.observed .>= notif.lower) .&
                (notif.observed .<= notif.upper)
        ), digits=3),
    )

    println("\nNotification residual autocorrelation:")
    show(stdout, MIME"text/plain"(), acf; allrows=true)
    println()

    println("\nMean absolute fraction errors:")
    println(
        "Recurrence: ",
        round(mean(abs.(rec.observed .- rec.predicted)), digits=4),
    )
    println(
        "Pulmonary: ",
        round(mean(abs.(site.observed .- site.predicted)), digits=4),
    )

    println("\nResults: ", dir)

    return (
        notifications=notif,
        recurrence=rec,
        site=site,
        model=model_df,
        who=who,
        acf=acf,
    )
end
