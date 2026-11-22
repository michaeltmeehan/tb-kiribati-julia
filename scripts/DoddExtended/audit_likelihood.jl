
using TBKiribatiJulia
using DataFrames, CSV, Statistics
using Distributions: Normal

function audit_likelihood(fit; orders=(20, 50, 100, 200))
    TB = TBKiribatiJulia
    problem = fit.problem
    theta = fit.parameters
    obsmodel = problem.observation_model

    @assert obsmodel isa TB.AnnualRandomEffectNB2Observation

    # Simulate the epidemic exactly once.
    out = predict(fit)
    rows = TB.notification_fit_rows(problem, out)

    grouped = [
        filter(r -> r.year == obs.year, rows)
        for obs in problem.observations
    ]

    # --------------------------------------------------
    # A. Verify likelihood decomposition
    # --------------------------------------------------

    lnotif = TB.notification_loglikelihood(
        obsmodel, problem.observations, out,
        problem.age_breaks, theta,
    )
    lrec = TB.recurrence_loglikelihood(problem.recurrence, out)
    lsite = TB.site_loglikelihood(problem.site, out)
    ldeath = TB.treatment_death_loglikelihood(
        problem.treatment_deaths, theta
    )

    reconstructed = lnotif + lrec + lsite + ldeath
    direct = TB.calibration_loglikelihood(
        problem, theta; output=out
    )

    println("\n1. Likelihood consistency")
    println("-------------------------")
    println("Notifications:  ", lnotif)
    println("Recurrence:     ", lrec)
    println("Site:           ", lsite)
    println("Deaths:         ", ldeath)
    println("Reconstructed:  ", reconstructed)
    println("Direct:         ", direct)
    println("Stored fit:     ", fit.log_likelihood)

    @assert isapprox(reconstructed, direct; atol=1e-7)
    @assert isapprox(direct, fit.log_likelihood; atol=1e-5)

    # --------------------------------------------------
    # B. Quadrature convergence (no new ODE simulations)
    # --------------------------------------------------

    qrows = NamedTuple[]

    for n in orders
        obs = TB.AnnualRandomEffectNB2Observation(
            obsmodel.sigma_prior,
            obsmodel.tau_prior;
            quadrature_order=n,
        )

        value = sum(
            TB.observation_year_loglikelihood(obs, rr, theta)
            for rr in grouped
        )

        push!(qrows, (
            order=n,
            loglikelihood=value,
        ))
    end

    quadrature = DataFrame(qrows)
    reference = quadrature.loglikelihood[end]
    quadrature.difference =
        quadrature.loglikelihood .- reference

    println("\n2. Quadrature convergence")
    println("-------------------------")
    show(stdout, MIME"text/plain"(), quadrature; allrows=true)
    println()

    # --------------------------------------------------
    # C. Conditional annual multipliers
    # --------------------------------------------------

    # Use highest-order quadrature as the reference.
    reference_model = TB.AnnualRandomEffectNB2Observation(
        obsmodel.sigma_prior,
        obsmodel.tau_prior;
        quadrature_order=last(orders),
    )

    sigma = theta.sigma_obs
    tau = theta.tau_year

    multipliers = exp.(
        tau .* reference_model.nodes .- tau^2 / 2
    )

    results = NamedTuple[]

    for (obs, rr) in zip(problem.observations, grouped)
        logweights = [
            log(reference_model.weights[k]) +
                sum(
                TB.nb2_loglikelihood(
                    r.observed,
                    r.expected * multipliers[k],
                    sigma,
                )
                for r in rr
            )
            for k in eachindex(multipliers)
        ]

        maximum_logweight = maximum(logweights)
        weights = exp.(logweights .- maximum_logweight)
        weights ./= sum(weights)

        mean_multiplier = sum(weights .* multipliers)

        observed = sum(r.observed for r in rr)
        predicted = sum(r.expected for r in rr)

        push!(results, (
            year=obs.year,
            observed=observed,
            predicted=predicted,
            observed_ratio=observed / predicted,
            mean_multiplier=mean_multiplier,
            conditional_mean=predicted * mean_multiplier,
            remaining_residual=observed -
                predicted * mean_multiplier,
            annual_loglikelihood=
            TB.observation_year_loglikelihood(
                reference_model, rr, theta
            ),
            n_categories=length(rr),
        ))
    end

    annual = DataFrame(results)

    println("\n3. Annual multiplier diagnostics")
    println("--------------------------------")
    display(annual)

    println("\n4. Summary")
    println("-----------")
    println("Sigma observation: ", sigma)
    println("Tau annual:        ", tau)

    println(
        "Mean multiplier: ",
        mean(annual.mean_multiplier),
    )

    println(
        "Median multiplier: ",
        median(annual.mean_multiplier),
    )

    println(
        "Years multiplier > 2: ",
        count(>(2), annual.mean_multiplier),
    )

    println(
        "Years multiplier < 0.5: ",
        count(<(0.5), annual.mean_multiplier),
    )

    # Serial correlation of conditional annual effects.
    m = log.(annual.mean_multiplier)

    for lag in (1, 2, 3)
        if length(m) > lag + 2
            rho = cor(m[1:(end-lag)], m[(1+lag):end])
            println("Log multiplier ACF($lag): ", rho)
        end
    end

    # Compare observed / marginal-predicted totals.
    println(
        "Overall observed/predicted ratio: ",
        sum(annual.observed) / sum(annual.predicted),
    )

    # --------------------------------------------------
    # D. Save results
    # --------------------------------------------------

    dir = joinpath(
        @__DIR__, "..", "..", "results",
        "DoddExtended", "initial_calibration",
    )

    mkpath(dir)

    CSV.write(joinpath(dir, "likelihood_audit_annual.csv"), annual)
    CSV.write(joinpath(dir, "likelihood_audit_quadrature.csv"), quadrature)

    return (
        annual=annual,
        quadrature=quadrature,
        output=out,
    )
end
