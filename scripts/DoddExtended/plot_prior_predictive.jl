using TBKiribatiJulia
using CSV, DataFrames
using CairoMakie
using Random
using Statistics

include(joinpath(
    @__DIR__, "..", "common", "kiribati_paths.jl"
))

let
    TB = TBKiribatiJulia

    root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = kiribati_data_paths(root)

    prior_dir = joinpath(
        root, "results", "DoddExtended",
        "prior_predictive"
    )

    posterior_dir = joinpath(
        root, "results", "DoddExtended",
        "posterior_epidemiology"
    )

    figure_dir = joinpath(prior_dir, "figures")
    mkpath(figure_dir)

    age_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+"
    ]

    # --------------------------------------------------
    # Load simulation outputs
    # --------------------------------------------------

    prior = CSV.read(
        joinpath(prior_dir, "draws_annual.csv"),
        DataFrame
    )

    age = CSV.read(
        joinpath(prior_dir, "draws_age_2023.csv"),
        DataFrame
    )

    failures = CSV.read(
        joinpath(prior_dir, "failures.csv"),
        DataFrame
    )

    failed_ids = Set(failures.draw)

    filter!(r -> !(r.draw in failed_ids), prior)
    filter!(r -> !(r.draw in failed_ids), age)

    # Derived outcomes
    prior.detection_ratio =
        prior.notifications_mean ./ prior.incidence

    age.incidence_per_100k =
        1e5 .* age.incidence ./ age.population

    # --------------------------------------------------
    # Observations
    # --------------------------------------------------

    surveillance = TB.load_kiribati_surveillance(
        paths.surveillance;
        age_labels,
        age_years=2013:2024,
    )

    years = Set(prior.year)

    national = DataFrame(
        surveillance.national_notifications
    )

    recurrent = DataFrame(
        surveillance.recurrent_notifications
    )

    site = DataFrame(
        surveillance.site_notifications
    )

    filter!(r -> r.year in years, national)
    filter!(r -> r.year in years, recurrent)
    filter!(r -> r.year in years, site)

    # Recurrent fraction
    recurrence = innerjoin(
        rename(recurrent, :cases => :recurrent),
        rename(national, :cases => :total);
        on=:year,
    )

    recurrence.value =
        recurrence.recurrent ./ recurrence.total

    # Pulmonary fraction
    site.value = site.pulmonary ./ site.total

    observations = Dict(
        :notifications_predictive =>
            select(national, :year, :cases => :value),
        :notifications_mean =>
            select(national, :year, :cases => :value),
        :recurrent_fraction =>
            select(recurrence, :year, :value),
        :pulmonary_fraction =>
            select(site, :year, :value),
    )

    # --------------------------------------------------
    # WHO burden estimates (external comparisons)
    # --------------------------------------------------

    who = CSV.read(
        joinpath(
            paths.estimates,
            "tb_burden_national.csv",
        ),
        DataFrame,
    )

    filter!(r -> r.year in years, who)

    who_metrics = (
        incidence=(
            :e_inc_num, :e_inc_num_lo, :e_inc_num_hi
        ),
        mortality=(
            :e_mort_num, :e_mort_num_lo, :e_mort_num_hi
        ),
        detection_ratio=(
            :c_cdr, :c_cdr_lo, :c_cdr_hi
        ),
    )

    for (metric, cols) in pairs(who_metrics)
        obs = select(
            who,
            :year,
            cols[1] => :value,
            cols[2] => :lower,
            cols[3] => :upper,
        )

        dropmissing!(obs)

        # WHO case detection ratio is reported as a percentage.
        # Convert to a proportion for consistency with model outputs.
        if metric == :detection_ratio
            for col in (:value, :lower, :upper)
                obs[!, col] = obs[!, col] ./ 100
            end
        end

        obs.source = fill("WHO estimate", nrow(obs))

        observations[metric] = obs
    end

    # --------------------------------------------------
    # Conditional binomial predictive distributions
    # --------------------------------------------------

    rng = MersenneTwister(20261126)

    # Observed recurrence denominators

    recurrence_obs = select(
        recurrence,
        :year,
        :recurrent => :numerator,
        :total => :denominator,
    )

    # Observed site-classified denominators

    pulmonary_obs = select(
        site,
        :year,
        :pulmonary => :numerator,
        :total => :denominator,
    )

    # Treatment cohorts: same years as MCMC likelihood

    treatment_obs = DataFrame(
        TB.TreatmentDeathObservation(
            filter(
                r -> 2012 <= r.year <= 2023,
                surveillance.treatment_outcomes,
            )
        ).rows
    )

    treatment_obs = select(
        treatment_obs,
        :year,
        :died => :numerator,
        :cohort_size => :denominator,
    )

    # Generate replicated observations

    recurrence_pred = TB.conditional_binomial_predictive(
        prior,
        recurrence_obs,
        :recurrent_fraction;
        rng,
    )

    pulmonary_pred = TB.conditional_binomial_predictive(
        prior,
        pulmonary_obs,
        :pulmonary_fraction;
        rng,
    )

    # Recover the fixed treatment CFR used in prior simulations

    q_T = median(TB.doddextended_priors().treatment_cfr)

    fixed_overrides = TB.extended_fixed_parameters()

    if haskey(fixed_overrides, :treatment_cfr)
        q_T = fixed_overrides.treatment_cfr
    end

    # Treatment predictions require no ODE outputs

    treatment_draws = crossjoin(
        DataFrame(draw=unique(prior.draw)),
        select(treatment_obs, :year),
    )

    treatment_draws.treatment_cfr =
        fill(q_T, nrow(treatment_draws))

    treatment_pred = TB.conditional_binomial_predictive(
        treatment_draws,
        treatment_obs,
        :treatment_cfr;
        rng,
    )

    # Save predictive replicates

    for (name, df) in (
        ("recurrence", recurrence_pred),
        ("pulmonary", pulmonary_pred),
        ("treatment_deaths", treatment_pred),
    )
        CSV.write(
            joinpath(
                prior_dir,
                "predictive_$name.csv"
            ),
            df,
        )
    end

    # --------------------------------------------------
    # Labels and scales
    # --------------------------------------------------

    labels = Dict(
        :incidence => "Annual TB incidence",
        :notifications_mean => "Annual TB notifications",
        :notifications_predictive => "Annual TB notifications",
        :mortality => "Annual TB deaths",
        :detection_ratio => "Notification-to-incidence ratio (%)",
        :recurrent_fraction => "Recurrent notifications (%)",
        :pulmonary_fraction => "Pulmonary notifications (%)",
        :incidence_per_100k => "TB incidence per 100,000",
    )

    scales = Dict(
        :detection_ratio => 100.0,
        :recurrent_fraction => 100.0,
        :pulmonary_fraction => 100.0,
    )

    # Figure-saving helper
    function save_figure(name, fig)
        display(fig)

        save(
            joinpath(figure_dir, "$name.png"),
            fig;
            px_per_unit=3,
        )

        save(
            joinpath(figure_dir, "$name.svg"),
            fig,
        )
    end

    # --------------------------------------------------
    # Conditional binomial predictive checks
    # --------------------------------------------------

    conditional_draws = Dict(
        :recurrence => recurrence_pred,
        :pulmonary => pulmonary_pred,
        :treatment_deaths => treatment_pred,
    )

    conditional_labels = Dict(
        :recurrence => "Recurrent notifications (%)",
        :pulmonary => "Pulmonary notifications (%)",
        :treatment_deaths => "Treatment cohort deaths (%)",
    )

    fig = Figure(size=(1350, 430))

    for (i, metric) in enumerate(
        (:recurrence, :pulmonary, :treatment_deaths)
    )
        d = conditional_draws[metric]

        ax = Axis(
            fig[1, i];
            title=conditional_labels[metric],
            xlabel="Year",
            xticks=2012:3:2024,
        )

        expected = TB.predictive_intervals(
            d, :expected_fraction
        )

        predictive = TB.predictive_intervals(
            d, :predictive_fraction
        )

        # Full predictive uncertainty

        band!(
            ax,
            predictive.group,
            100 .* predictive.q025,
            100 .* predictive.q975;
            color=(:steelblue, 0.18),
            label="Predictive 95%",
        )

        # Parameter uncertainty alone

        band!(
            ax,
            expected.group,
            100 .* expected.q025,
            100 .* expected.q975;
            color=(:steelblue, 0.35),
            label="Parameter 95%",
        )

        lines!(
            ax,
            expected.group,
            100 .* expected.median;
            color=:steelblue,
            linewidth=2,
            label="Prior median",
        )

        obs = unique(
            select(d, :year, :observed_fraction)
        )

        scatter!(
            ax,
            obs.year,
            100 .* obs.observed_fraction;
            color=:black,
            markersize=7,
            label="Observed",
        )

        ylims!(ax, 0, nothing)

        if i == 1
            axislegend(ax; position=:rt)
        end
    end

    save_figure("prior_predictive_conditional", fig)

    # --------------------------------------------------
    # Figure 1: National prior predictive analysis
    # --------------------------------------------------

    metrics = [
        :incidence,
        :notifications_predictive,
        :mortality,
        :detection_ratio,
        :recurrent_fraction,
        :pulmonary_fraction,
    ]

    fig = TB.plot_predictive_summary(
        prior, metrics;
        labels,
        scales,
        observations,
        ncols=2,
        prediction_label="Prior median",
        log_metrics=[
            :incidence,
            :notifications_predictive,
            :mortality,
        ],
    )

    save_figure("prior_predictive_national", fig)

    # --------------------------------------------------
    # Figure 2: Age-specific predictions for 2023
    # --------------------------------------------------

    age_obs = only(filter(
        r -> r.year == 2023,
        surveillance.age_notifications,
    ))

    age_observations = Dict(
        :notifications_predictive => DataFrame(
            age_group=age_obs.labels,
            value=age_obs.counts,
        )
    )

    fig = TB.plot_predictive_by_age(
        age,
        [
            :notifications_predictive,
            :incidence_per_100k,
        ];
        age_labels,
        labels,
        observations=age_observations,
        prediction_label="Prior median",
    )

    save_figure("prior_predictive_age_2023", fig)

    # --------------------------------------------------
    # Figure 3: Effect of observation uncertainty
    # --------------------------------------------------

    expected = select(
        prior,
        :year,
        :notifications_mean => :notifications,
    )

    predictive = select(
        prior,
        :year,
        :notifications_predictive => :notifications,
    )

    fig = TB.plot_predictive_comparison(
        expected,
        predictive,
        [:notifications];
        labels=Dict(
            :notifications => "Annual TB notifications"
        ),
        observations=Dict(
            :notifications =>
                observations[:notifications_mean]
        ),
        series_labels=(
            "Parameter uncertainty",
            "Including observation uncertainty",
        ),
        ncols=1,
    )

    save_figure("prior_observation_uncertainty", fig)

    # --------------------------------------------------
    # Figure 4: Prior versus posterior predictions
    # --------------------------------------------------

    posterior_file = joinpath(
        posterior_dir, "annual_draws.csv"
    )

    if isfile(posterior_file)
        posterior = CSV.read(
            posterior_file, DataFrame
        )

        posterior_failures = CSV.read(
            joinpath(posterior_dir, "failed_draws.csv"),
            DataFrame,
        )

        failed = Set(posterior_failures.draw_id)

        filter!(
            r -> !(r.draw_id in failed),
            posterior,
        )

        # Harmonise prior and posterior column names
        rename!(
            posterior,
            :notifications => :notifications_mean,
            :tb_deaths => :mortality,
        )

        posterior.detection_ratio =
            posterior.notifications_mean ./
                posterior.incidence

        # Restrict comparisons to common years
        common_years = intersect(
            unique(prior.year),
            unique(posterior.year),
        )

        prior_common = filter(
            r -> r.year in common_years, prior
        )

        posterior_common = filter(
            r -> r.year in common_years, posterior
        )

        comparison_metrics = [
            :incidence,
            :notifications_mean,
            :mortality,
            :detection_ratio,
            :recurrent_fraction,
            :pulmonary_fraction,
        ]

        fig = TB.plot_predictive_comparison(
            prior_common,
            posterior_common,
            comparison_metrics;
            labels,
            scales,
            observations,
            ncols=2,
        )

        save_figure("prior_posterior_epidemiology", fig)

    else
        @warn "Posterior predictions not found"
    end

    println("Figures saved to: $figure_dir")
end