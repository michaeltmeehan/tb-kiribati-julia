using TBKiribatiJulia
using CSV, DataFrames
using Distributions
using CairoMakie

let
    TB = TBKiribatiJulia

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    output_dir = normpath(joinpath(
        @__DIR__, "..", "..",
        "results", "DoddExtended",
        "six_parameter_turing"
    ))

    posterior_file = joinpath(output_dir, "posterior_draws.csv")
    prior_file = joinpath(output_dir, "prior_parameters.csv")
    figure_dir = joinpath(output_dir, "figures")

    isfile(posterior_file) ||
        error("Posterior draws not found: $posterior_file")

    isfile(prior_file) ||
        error("Prior parameter file not found: $prior_file")

    mkpath(figure_dir)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    function reconstruct_distribution(
        dist_name::AbstractString,
        p1,
        p2,
    )
        if dist_name == "LogNormal"
            return LogNormal(p1, p2)
        elseif dist_name == "Normal"
            return Normal(p1, p2)
        elseif dist_name == "Beta"
            return Beta(p1, p2)
        elseif dist_name == "Gamma"
            return Gamma(p1, p2)
        else
            error("Unsupported distribution type: $dist_name")
        end
    end

    function build_prior_namedtuple(
        prior_table::DataFrame
    )
        pairs = Pair{Symbol,Distribution}[]

        for row in eachrow(prior_table)
            name = Symbol(row.parameter)
            dist = reconstruct_distribution(
                row.distribution,
                row.param1,
                row.param2,
            )
            push!(pairs, name => dist)
        end

        return (; pairs...)
    end

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    posterior = CSV.read(posterior_file, DataFrame)
    prior_table = CSV.read(prior_file, DataFrame)

    required_prior_cols = [
        :parameter,
        :distribution,
        :param1,
        :param2,
    ]

    for col in required_prior_cols
        col in names(prior_table) ||
            error("Missing column in prior table: $col")
    end

    prior = build_prior_namedtuple(prior_table)

    # --------------------------------------------------
    # Parameters and labels
    # --------------------------------------------------

    preferred_order = [
        :beta,
        :Lambda0,
        :nu,
        :omega,
        :detection_logitK,
        :r_SP,
    ]

    parameters = [
        p for p in preferred_order
              if p in keys(prior) && p in propertynames(posterior)
    ]

    isempty(parameters) &&
        error("No overlapping parameters found between prior and posterior files.")

    for p in parameters
        all(isfinite, posterior[!, p]) ||
            error("Non-finite posterior values found for parameter $p")
    end

    labels = Dict(
        :beta => "Transmission coefficient (β)",
        :Lambda0 => "Initial force of infection (Λ₀)",
        :nu => "Late progression rate (ν)",
        :omega => "Relapse rate (ω)",
        :detection_logitK => "Detection logit intercept",
        :r_SP => "SP detection multiplier",
    )

    log_parameters = [
        :beta,
        :Lambda0,
        :nu,
        :omega,
        :r_SP,
    ]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig = TB.plot_prior_posterior(
        prior,
        posterior;
        parameters=parameters,
        labels=labels,
        log_parameters=log_parameters,
        ncols=3,
    )

    display(fig)

    save(
        joinpath(figure_dir, "prior_posterior.png"),
        fig;
        px_per_unit=3,
    )

    save(
        joinpath(figure_dir, "prior_posterior.svg"),
        fig,
    )

    # --------------------------------------------------
    # Optional summary table
    # --------------------------------------------------

    summary_table = DataFrame(
        parameter=String[],
        posterior_median=Float64[],
        posterior_q025=Float64[],
        posterior_q975=Float64[],
    )

    for p in parameters
        x = collect(skipmissing(posterior[!, p]))
        push!(summary_table, (
            string(p),
            quantile(x, 0.5),
            quantile(x, 0.025),
            quantile(x, 0.975),
        ))
    end

    CSV.write(
        joinpath(output_dir, "posterior_summary.csv"),
        summary_table,
    )
end