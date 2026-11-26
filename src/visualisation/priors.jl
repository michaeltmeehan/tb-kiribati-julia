using CairoMakie
using Distributions

"""
    plot_prior_posterior(prior, posterior=nothing; kwargs...)

Plot analytical prior densities and optionally overlay
posterior marginal density estimates.

Posterior samples should be supplied as a table with
columns corresponding to parameter names.
"""
function plot_prior_posterior(
    prior::NamedTuple,
    posterior=nothing;
    parameters=collect(keys(prior)),
    labels=Dict(),
    log_parameters=Symbol[],
    ncols=3,
    tail_probability=0.001,
)
    names = collect(parameters)
    nrows = cld(length(names), ncols)

    fig = Figure(
        size=(350 * ncols, 260 * nrows + 50)
    )

    for (i, name) in enumerate(names)

        dist = getproperty(prior, name)

        lo = quantile(dist, tail_probability)
        hi = quantile(dist, 1 - tail_probability)

        use_log = name in log_parameters

        ax = Axis(
            fig[cld(i, ncols), mod1(i, ncols)];
            xlabel=get(labels, name, string(name)),
            ylabel="Density",
        )

        if use_log
            x = range(log10(lo), log10(hi); length=500)
            y = pdf.(dist, 10 .^ x) .*
                log(10) .* 10 .^ x

            ax.xlabel = "log10(" *
                get(labels, name, string(name)) * ")"
        else
            x = range(lo, hi; length=500)
            y = pdf.(dist, x)
        end

        band!(ax, x, zeros(length(x)), y;
            color=(:grey, 0.2))

        lines!(ax, x, y;
            color=:grey40,
            linewidth=2,
            label="Prior")

        if !isnothing(posterior)
            samples = Float64.(
                collect(skipmissing(
                    getproperty(posterior, name)
                ))
            )

            values = display_values(
                samples, name, log_parameters
            )

            if length(unique(values)) >= 2
                density!(ax, values;
                    color=(:steelblue, 0.25),
                    strokecolor=:steelblue,
                    strokewidth=2,
                    label="Posterior")
            elseif !isempty(values)
                vlines!(ax, [first(values)];
                    color=:steelblue,
                    linewidth=2,
                    label="Fixed value")
            end
        end

        if i == 1
            axislegend(ax; position=:rt)
        end
    end

    return fig
end


function plot_prior_posterior(
    problem::CalibrationProblem,
    posterior=nothing;
    kwargs...
)
    return plot_prior_posterior(
        problem.prior, posterior; kwargs...
    )
end