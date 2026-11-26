using TBKiribatiJulia
using CSV, DataFrames
using CairoMakie
using Random, Statistics

include(joinpath(
    @__DIR__, "..", "common", "kiribati_paths.jl"
))

let
    TB = TBKiribatiJulia

    root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = kiribati_data_paths(root)

    output_dir = joinpath(
        root, "results", "DoddExtended",
        "posterior_epidemiology"
    )

    figure_dir = joinpath(output_dir, "figures")
    mkpath(figure_dir)

    rng = MersenneTwister(20261126)

    age_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+"
    ]

    # --------------------------------------------------
    # Load posterior predictions
    # --------------------------------------------------

    annual = CSV.read(
        joinpath(output_dir, "annual_draws.csv"),
        DataFrame,
    )

    age = CSV.read(
        joinpath(output_dir, "age_notification_draws.csv"),
        DataFrame,
    )

    failures = CSV.read(
        joinpath(output_dir, "failed_draws.csv"),
        DataFrame,
    )

    failed = Set(failures.draw_id)

    filter!(r -> !(r.draw_id in failed), annual)
    filter!(r -> !(r.draw_id in failed), age)

    # --------------------------------------------------
    # Load observations
    # --------------------------------------------------

    surveillance = TB.load_kiribati_surveillance(
        paths.surveillance;
        age_labels,
        age_years=2013:2024,
    )

    national = DataFrame(
        surveillance.national_notifications
    )

    recurrent = DataFrame(
        surveillance.recurrent_notifications
    )

    site = DataFrame(
        surveillance.site_notifications
    )

    filter!(r -> 1980 <= r.year <= 2025, national)
    filter!(r -> 1996 <= r.year <= 2024, recurrent)
    filter!(r -> 1970 <= r.year <= 2025, site)

    # --------------------------------------------------
    # Figure-saving helper
    # --------------------------------------------------

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
    # Replicate notification observations
    # --------------------------------------------------

    pred = TB.notification_predictive(
        annual,
        age;
        sigma_obs=0.30,
        tau_year=0.18,
        rng,
    )

    annual_pred = pred.annual
    age_pred = pred.age

    # Add national observations for years without
    # age-disaggregated observations.

    national_lookup = Dict(
        r.year => r.cases
        for r in eachrow(national)
    )

    annual_pred.observed = [
        ismissing(r.observed) ?
        get(national_lookup, r.year, missing) :
        r.observed
        for r in eachrow(annual_pred)
    ]

    annual_pred.group = fill(
        "Total TB notifications",
        nrow(annual_pred),
    )

    CSV.write(
        joinpath(output_dir, "predictive_notifications.csv"),
        annual_pred,
    )

    CSV.write(
        joinpath(output_dir, "predictive_notifications_age.csv"),
        age_pred,
    )

    # --------------------------------------------------
    # Figure 1: National notification fit
    # --------------------------------------------------

    fig = TB.plot_predictive_groups(
        annual_pred;
        group=:group,
        ncols=1,
        ylabel="Annual notifications",
        year_step=5,
    )

    # Make national time-series figure wider
    resize!(fig.scene, (1000, 450))

    save_figure("posterior_predictive_national", fig)

    # --------------------------------------------------
    # Figure 2: Age-specific notification fits
    # --------------------------------------------------

    # Include combined 0-14 group where available.
    preferred_order = [
        "0-4", "5-14", "0-14", "15-24",
        "25-34", "35-44", "45-54",
        "55-64", "65+",
    ]

    available = unique(age_pred.age_group)

    group_order = [
        a for a in preferred_order
              if a in available
    ]

    append!(
        group_order,
        setdiff(available, group_order),
    )

    fig = TB.plot_predictive_groups(
        age_pred;
        group=:age_group,
        group_order,
        ncols=3,
        ylabel="Notifications",
        year_step=3,
    )

    save_figure("posterior_predictive_age", fig)

    # --------------------------------------------------
    # Conditional binomial observations
    # --------------------------------------------------

    recurrence_obs = innerjoin(
        rename(recurrent, :cases => :numerator),
        rename(national, :cases => :denominator);
        on=:year,
    )

    recurrence_obs = select(
        recurrence_obs,
        :year, :numerator, :denominator,
    )

    pulmonary_obs = select(
        site,
        :year,
        :pulmonary => :numerator,
        :total => :denominator,
    )

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

    # Harmonise draw identifiers
    conditional = rename(
        annual,
        :draw_id => :draw,
    )

    recurrence_pred = TB.conditional_binomial_predictive(
        conditional,
        recurrence_obs,
        :recurrent_fraction;
        rng,
    )

    pulmonary_pred = TB.conditional_binomial_predictive(
        conditional,
        pulmonary_obs,
        :pulmonary_fraction;
        rng,
    )

    # --------------------------------------------------
    # Fixed treatment cohort mortality
    # --------------------------------------------------

    fixed_file = joinpath(
        root, "results", "DoddExtended",
        "six_parameter_turing",
        "fixed_parameters.csv",
    )

    if isfile(fixed_file)

        fixed = CSV.read(fixed_file, DataFrame)

        q = only(
            fixed.value[
                fixed.parameter .== "treatment_cfr"
            ]
        )

        q_T = q isa Real ?
              Float64(q) : parse(Float64, q)

    else

        @warn "Using current fixed treatment CFR definitions"

        q_T = median(
            TB.doddextended_priors().treatment_cfr
        )

        overrides = TB.extended_fixed_parameters()

        if haskey(overrides, :treatment_cfr)
            q_T = overrides.treatment_cfr
        end
    end

    treatment_draws = crossjoin(
        DataFrame(draw=unique(conditional.draw)),
        select(treatment_obs, :year),
    )

    treatment_draws.treatment_cfr = fill(
        q_T, nrow(treatment_draws)
    )

    treatment_pred = TB.conditional_binomial_predictive(
        treatment_draws,
        treatment_obs,
        :treatment_cfr;
        rng,
    )

    # --------------------------------------------------
    # Figure 3: Conditional binomial predictive checks
    # --------------------------------------------------

    conditional_tables = DataFrame[]

    for (name, d) in (
        ("Recurrent notifications", recurrence_pred),
        ("Pulmonary notifications", pulmonary_pred),
        ("Treatment cohort deaths", treatment_pred),
    )

        tmp = select(
            d,
            :draw,
            :year,
            :expected_fraction => :expected,
            :predictive_fraction => :predictive,
            :observed_fraction => :observed,
        )

        tmp.group = fill(name, nrow(tmp))

        push!(conditional_tables, tmp)
    end

    conditional_pred = reduce(vcat, conditional_tables)

    CSV.write(
        joinpath(output_dir, "predictive_conditional.csv"),
        conditional_pred,
    )

    fig = TB.plot_predictive_groups(
        conditional_pred;
        group=:group,
        ncols=3,
        scale=100.0,
        ylabel="Percentage",
        year_step=5,
    )

    save_figure(
        "posterior_predictive_conditional",
        fig,
    )

    println("Figures saved to: $figure_dir")
end