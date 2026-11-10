# tb_notification_temporal_trends.jl
# -----------------------------------------------------------------------------
# Age-specific temporal trends in TB notifications, Kiribati, 2013–2023
#
# Run after the existing project has defined:
#   get_population(year)                      -> population by single year of age
#   aggregate_age_groups(population, breaks)  -> population grouped by AGE_BREAKS
#
# From the project's scripts directory (with the functions already loaded):
#   include("scripts/tb_notification_temporal_trends.jl")
#
# Packages: CSV, DataFrames, GLM, Distributions, CairoMakie, Statistics, Printf
#
# Response: observed notifications (counts), NOT the precomputed rates.
# Offset:   log(age-group-specific population in each calendar year).
# Mean:     E[cases_{a,t}] = population_{a,t} * exp(alpha_a + beta_a*(t-2018)).
#
# Quasi-Poisson inference is implemented using a Poisson GLM for the mean
# and estimating phi = sum((y-mu)^2 / mu) / (n-p). Coefficient covariance is
# multiplied by max(1, phi): a conservative guard against sub-Poisson scale.
# Student-t reference distributions are used for coefficients and a quasi-
# deviance approximate F test for comparisons of nested models.
#
# Quasi-Poisson is not a full probability distribution: do not use ordinary
# Poisson p-values, Poisson AIC, or Poisson likelihood-ratio tests to interpret
# these overdispersed fits. CIs below concern the fitted MEAN RATE, not a
# prediction interval for future annual observed rates.
# -----------------------------------------------------------------------------

using CSV
using DataFrames
using Statistics
using GLM
using Distributions
using CairoMakie
using Printf
using LinearAlgebra


# Benjamini–Hochberg adjustment of eight exploratory age-specific slope tests.
function tbtrend_bh(pvalues)
    p = Float64.(pvalues)
    n = length(p)
    order = sortperm(p)
    adjusted = similar(p)
    running = 1.0
    for rank in n:-1:1
        idx = order[rank]
        running = min(running, n * p[idx] / rank)
        adjusted[idx] = min(1.0, running)
    end
    return adjusted
end


# Assemble the actual observed counts and age-year-specific population sizes.
function tbtrend_build_data(notifications, age_breaks, age_labels, years)
    rows = NamedTuple[]

    for year in years
        # The original exploratory script also passes years as Float64.
        pop = get_population(Float64(year))
        pop_grouped = Float64.(aggregate_age_groups(pop, age_breaks))
        length(pop_grouped) == length(age_labels) ||
            error("Population aggregation yielded $(length(pop_grouped)) groups in $year; expected $(length(age_labels)).")
        all(x -> isfinite(x) && x > 0, pop_grouped) ||
            error("Population must be positive and finite in every age group ($year).")

        year_rows = notifications[notifications.year .== year, :]
        for (g, label) in enumerate(age_labels)
            matches = year_rows[year_rows.age_group .== label, :cases]
            length(matches) == 1 ||
                error("Expected exactly one notification count for year=$year, age=$label; found $(length(matches)).")
            c = only(matches)
            ismissing(c) && error("Missing notification count for year=$year, age=$label.")
            isfinite(c) && c >= 0 && isinteger(c) ||
                error("Non-negative integer notification count required for year=$year, age=$label: $c")

            exposure = pop_grouped[g]
            push!(rows, (
                year = Int(year),
                age_group = label,
                age_index = g,
                t = Float64(year - 2018),
                cases = Int(c),
                population = exposure,
                observed_rate = 1e5 * c / exposure,
            ))
        end
    end

    df = DataFrame(rows)
    sort!(df, [:age_index, :year])
    return df
end


# Fit the mean structure with a Poisson GLM, then correct inference for
# overdispersion using a quasi-Poisson scale estimate.
function tbtrend_qpoisson(X, counts, population)
    x = Matrix{Float64}(X)
    y = Float64.(counts)
    exposure = Float64.(population)
    n, p = size(x)
    n > p || error("Insufficient residual degrees of freedom: n=$n, p=$p.")
    all(z -> isfinite(z) && z > 0, exposure) || error("Nonpositive population exposure.")

    offset = log.(exposure)
    model = GLM.glm(x, y, Poisson(), LogLink(); offset=offset, maxiter=200)
    beta = Float64.(GLM.coef(model))
    mu = exp.(x * beta .+ offset)
    all(z -> isfinite(z) && z > 0, mu) || error("Invalid fitted count means.")

    df = n - p
    pearson_chisq = sum(((y .- mu) .^ 2) ./ mu)
    phi_pearson = pearson_chisq / df
    phi = max(1.0, phi_pearson)  # conservative: never shrink Poisson uncertainty
    V = Matrix{Float64}(GLM.vcov(model)) .* phi

    return (
        model = model,
        beta = beta,
        vcov = V,
        mu = mu,
        df = df,
        dispersion_pearson = phi_pearson,
        dispersion_used = phi,
        deviance = GLM.deviance(model),
        pearson_chisq = pearson_chisq,
    )
end


# Student-t inference with dispersion-adjusted standard errors.
function tbtrend_slope_summary(fit, k)
    b = fit.beta[k]
    se = sqrt(max(0.0, fit.vcov[k, k]))
    se > 0 || error("Nonpositive slope standard error.")
    td = TDist(fit.df)
    crit = quantile(td, 0.975)
    lo = b - crit * se
    hi = b + crit * se
    p = 2 * ccdf(td, abs(b / se))

    return (
        log_rate_ratio = b,
        se_log_rate_ratio = se,
        annual_rate_ratio = exp(b),
        annual_rr_lower95 = exp(lo),
        annual_rr_upper95 = exp(hi),
        annual_change_pct = 100 * expm1(b),
        annual_change_lower95_pct = 100 * expm1(lo),
        annual_change_upper95_pct = 100 * expm1(hi),
        decade_rate_ratio = exp(10 * b),
        p_value = p,
    )
end


function tbtrend_by_age(data, age_labels, analysis)
    summary_rows = NamedTuple[]
    pred_rows = NamedTuple[]
    fits = Dict{String, Any}()

    for label in age_labels
        d = data[data.age_group .== label, :]
        sort!(d, :year)
        n = nrow(d)
        X = hcat(ones(n), Float64.(d.t))
        fit = tbtrend_qpoisson(X, d.cases, d.population)
        fits[label] = fit
        stats = tbtrend_slope_summary(fit, 2)

        push!(summary_rows, merge((
            analysis = analysis,
            age_group = label,
            start_year = minimum(d.year),
            end_year = maximum(d.year),
            n_years = n,
            total_cases = sum(d.cases),
            dispersion_pearson = fit.dispersion_pearson,
            dispersion_used = fit.dispersion_used,
            residual_df = fit.df,
        ), stats))

        tdcrit = quantile(TDist(fit.df), 0.975)
        for i in 1:n
            x = Vector{Float64}(X[i, :])
            eta = dot(x, fit.beta)
            se_eta = sqrt(max(0.0, dot(x, fit.vcov * x)))
            rate = 1e5 * exp(eta)
            fitted_cases = fit.mu[i]
            pearson_resid = (d.cases[i] - fitted_cases) /
                sqrt(fit.dispersion_used * fitted_cases)

            push!(pred_rows, (
                analysis = analysis,
                year = d.year[i],
                age_group = label,
                cases = d.cases[i],
                population = d.population[i],
                observed_rate = d.observed_rate[i],
                fitted_cases = fitted_cases,
                fitted_rate = rate,
                fitted_rate_lower95 = 1e5 * exp(eta - tdcrit * se_eta),
                fitted_rate_upper95 = 1e5 * exp(eta + tdcrit * se_eta),
                scaled_pearson_residual = pearson_resid,
            ))
        end
    end

    summary = DataFrame(summary_rows)
    pred = DataFrame(pred_rows)
    return summary, pred, fits
end


# Three nested age-stratified mean models, with the same population offset:
#   M0:  age-specific intercepts, no year trends
#   M1:  age-specific intercepts, one shared annual trend
#   M2:  age-specific intercepts and age-specific annual trends
# We use a direct design matrix so coefficient indexing is unambiguous.
function tbtrend_joint_tests(data, age_labels)
    n = nrow(data)
    g = length(age_labels)
    A = zeros(n, g)
    AT = zeros(n, g)
    for i in 1:n
        k = data.age_index[i]
        A[i, k] = 1.0
        AT[i, k] = data.t[i]
    end

    m0 = tbtrend_qpoisson(A, data.cases, data.population)
    m1 = tbtrend_qpoisson(hcat(A, Float64.(data.t)), data.cases, data.population)
    m2 = tbtrend_qpoisson(hcat(A, AT), data.cases, data.population)

    function compare(reduced, full, label)
        q = length(full.beta) - length(reduced.beta)
        q > 0 || error("Model comparison has no additional terms.")
        # Approximate quasi-deviance F test. Use scale from the larger model.
        numerator = max(0.0, reduced.deviance - full.deviance) / q
        f_statistic = numerator / full.dispersion_used
        pvalue = ccdf(FDist(q, full.df), f_statistic)
        return (
            comparison = label,
            numerator_df = q,
            denominator_df = full.df,
            reduced_deviance = reduced.deviance,
            full_deviance = full.deviance,
            dispersion_pearson = full.dispersion_pearson,
            dispersion_used = full.dispersion_used,
            F_statistic = f_statistic,
            p_value = pvalue,
        )
    end

    tests = DataFrame([
        compare(m0, m1, "Any common linear year trend (M0 vs M1)"),
        compare(m1, m2, "Do annual trends differ by age? (M1 vs M2)"),
        compare(m0, m2, "Any age-specific linear year trends (M0 vs M2)"),
    ])

    common_slope = tbtrend_slope_summary(m1, g + 1)
    return tests, common_slope, (m0 = m0, m1 = m1, m2 = m2)
end


function tbtrend_plot_rates(predictions, age_labels, outdir)
    fig = Figure(size=(1360, 1500), fontsize=18)
    Label(fig[0, 1:2],
          "Kiribati TB notification rates: observed and quasi-Poisson trends (2013–2023)";
          fontsize=23, tellwidth=false)

    for (g, label) in enumerate(age_labels)
        row = div(g - 1, 2) + 1
        col = mod(g - 1, 2) + 1
        ax = Axis(fig[row, col];
                  title="Age $label", xlabel="Year",
                  ylabel="Notifications per 100,000",
                  xticks=2013:2:2023)
        sub = predictions[predictions.age_group .== label, :]
        sort!(sub, :year)
        years = Float64.(sub.year)

        CairoMakie.band!(ax, years, sub.fitted_rate_lower95,
                         sub.fitted_rate_upper95; color=(:steelblue, 0.20))
        CairoMakie.lines!(ax, years, sub.fitted_rate;
                          color=:steelblue, linewidth=3)
        CairoMakie.scatter!(ax, years, sub.observed_rate;
                            color=:black, markersize=11)
        CairoMakie.xlims!(ax, 2012.5, 2023.5)
    end

    Label(fig[5, 1:2],
          "Dots: observed rates  |  Blue: fitted log-linear rate  |  Shading: 95% CI for fitted mean";
          fontsize=15, tellwidth=false)
    save(joinpath(outdir, "rates_observed_and_fitted.png"), fig)
    return fig
end


function tbtrend_plot_slope_sensitivity(trends, age_labels, outdir)
    analyses = [
        ("all_2013_2023", "2013–2023", :steelblue, :circle, -0.21),
        ("through_2022", "2013–2022", :darkorange, :rect, 0.0),
        ("omit_2020_2021", "Excluding 2020–2021", :darkgreen, :diamond, 0.21),
    ]

    fig = Figure(size=(1200, 670), fontsize=18)
    ax = Axis(fig[1, 1];
              title="Estimated annual change in age-specific TB notification rates",
              ylabel="Annual change (%)", xlabel="Age group",
              xticks=(1:length(age_labels), age_labels))
    CairoMakie.hlines!(ax, [0.0]; color=:gray50, linestyle=:dash, linewidth=2)

    for (which, caption, col, marker, shift) in analyses
        d = trends[trends.analysis .== which, :]
        # The source is ordered in AGE_LABELS order, including sensitivity runs.
        xpos = Float64[]
        for a in d.age_group
            j = findfirst(==(a), age_labels)
            isnothing(j) && error("Unexpected age group in trend summary: $a")
            push!(xpos, j + shift)
        end
        for j in eachindex(xpos)
            CairoMakie.lines!(ax, [xpos[j], xpos[j]],
                              [d.annual_change_lower95_pct[j], d.annual_change_upper95_pct[j]];
                              color=col, linewidth=2)
        end
        CairoMakie.scatter!(ax, xpos, d.annual_change_pct;
                            color=col, marker=marker, markersize=16, label=caption)
    end
    CairoMakie.axislegend(ax; position=:rt)
    CairoMakie.xlims!(ax, 0.5, length(age_labels) + 0.5)
    save(joinpath(outdir, "annual_trends_sensitivity.png"), fig)
    return fig
end


function tbtrend_plot_residuals(predictions, age_labels, outdir)
    fig = Figure(size=(1100, 540), fontsize=17)
    ax = Axis(fig[1, 1];
              title="Scaled Pearson residuals: main 2013–2023 age-specific fits",
              xlabel="Year", ylabel="Age group",
              xticks=(2013:2023, string.(2013:2023)),
              yticks=(1:length(age_labels), age_labels))

    xs = Float64.(predictions.year)
    ys = Float64[findfirst(==(label), age_labels) for label in predictions.age_group]
    cs = Float64.(predictions.scaled_pearson_residual)
    bound = max(2.0, maximum(abs.(cs)))
    hm = CairoMakie.scatter!(ax, xs, ys;
                             color=cs, colormap=:balance, colorrange=(-bound, bound),
                             marker=:rect, markersize=29)
    Colorbar(fig[1, 2], hm; label="Scaled Pearson residual")
    save(joinpath(outdir, "scaled_pearson_residuals.png"), fig)
    return fig
end


function tbtrend_main()
    isdefined(Main, :get_population) || error(
        "Load the project's get_population(year) function before including this script.")
    isdefined(Main, :aggregate_age_groups) || error(
        "Load the project's aggregate_age_groups(population, breaks) function before including this script.")

    data_dir = normpath(joinpath(@__DIR__, "..", "data"))
    output_dir = normpath(joinpath(@__DIR__, "..", "results", "notification_temporal_trends"))
    mkpath(output_dir)

    age_breaks = [0, 5, 15, 25, 35, 45, 55, 65, 96]
    age_labels = ["0-4", "5-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    years = 2013:2023

    notifications = CSV.read(joinpath(data_dir, "tb_notifications_by_age.csv"), DataFrame)
    all(c -> c in propertynames(notifications), [:year, :age_group, :cases]) ||
        error("Notification CSV must have year, age_group, cases columns.")

    data = tbtrend_build_data(notifications, age_breaks, age_labels, years)
    nrow(data) == length(years) * length(age_labels) || error("Unexpected number of observations.")
    CSV.write(joinpath(output_dir, "counts_and_population_offsets.csv"), data)

    analysis_sets = [
        ("all_2013_2023", data),
        ("through_2022", data[data.year .<= 2022, :]),
        ("omit_2020_2021", data[.!in.(data.year, Ref((2020, 2021))), :]),
    ]

    all_summaries = DataFrame[]
    full_predictions = nothing
    for (label, subset) in analysis_sets
        stats, predicted, _ = tbtrend_by_age(subset, age_labels, label)
        push!(all_summaries, stats)
        if label == "all_2013_2023"
            full_predictions = predicted
        end
    end
    trends = vcat(all_summaries...)
    trends[!, :p_bh_age8] = fill!(Vector{Union{Missing, Float64}}(undef, nrow(trends)), missing)
    idx = findall(trends.analysis .== "all_2013_2023")
    trends[idx, :p_bh_age8] = tbtrend_bh(trends.p_value[idx])

    CSV.write(joinpath(output_dir, "age_specific_trends.csv"), trends)
    CSV.write(joinpath(output_dir, "main_fitted_rates_and_residuals.csv"), full_predictions)

    tests, common_trend, joint_fits = tbtrend_joint_tests(data, age_labels)
    CSV.write(joinpath(output_dir, "joint_model_tests.csv"), tests)

    tbtrend_plot_rates(full_predictions, age_labels, output_dir)
    tbtrend_plot_slope_sensitivity(trends, age_labels, output_dir)
    tbtrend_plot_residuals(full_predictions, age_labels, output_dir)

    println("\nTB notification trend analysis: Kiribati, 2013–2023")
    println("Counts modelled with log(population) offset; Pearson quasi-Poisson scale.")
    println("Phi used = max(1, Pearson χ² / residual df).")
    println("\nAge-specific annual rate changes, main 2013–2023 fit:")
    @printf("%-9s %12s %24s %12s %11s %11s\n",
            "Age", "Change %/yr", "95% CI (%)", "p", "BH p", "phi")
    main_rows = trends[trends.analysis .== "all_2013_2023", :]
    for row in eachrow(main_rows)
        @printf("%-9s %+11.2f%%  [%+8.2f, %+8.2f]  %11.4g %11.4g %11.2f\n",
                row.age_group, row.annual_change_pct,
                row.annual_change_lower95_pct, row.annual_change_upper95_pct,
                row.p_value, row.p_bh_age8, row.dispersion_used)
    end

    println("\nShared annual trend (pooled quasi-Poisson model, adjusted for age group):")
    @printf("  %+0.2f%% per year [95%% CI %+0.2f%% to %+0.2f%%]\n",
            common_trend.annual_change_pct,
            common_trend.annual_change_lower95_pct,
            common_trend.annual_change_upper95_pct)
    println("\nNested-model quasi-deviance approximate F tests:")
    for row in eachrow(tests)
        @printf("  %-49s F(%d, %d)=%.3f; p=%.4g\n", row.comparison,
                row.numerator_df, row.denominator_df, row.F_statistic, row.p_value)
    end
    println("\nOutputs: ", output_dir)
    println("  counts_and_population_offsets.csv")
    println("  age_specific_trends.csv (three analysis windows; BH p-values for main fit)")
    println("  main_fitted_rates_and_residuals.csv")
    println("  joint_model_tests.csv")
    println("  rates_observed_and_fitted.png")
    println("  annual_trends_sensitivity.png")
    println("  scaled_pearson_residuals.png")
    println("\nCaution: 11 years per age group; nonlinear trends, temporal autocorrelation,")
    println("observation-process changes and period effects remain possible. These CIs")
    println("are conditional on the denominators supplied by get_population().")
    return (data=data, trends=trends, predictions=full_predictions,
            tests=tests, common_trend=common_trend)
end

# Results are returned so they can be inspected interactively if desired.
tbtrend_results = tbtrend_main()
