
using CairoMakie
using StatsBase: corspearman

# Transform values for display, without modifying the
# underlying sensitivity data.
function display_values(values, name, log_parameters)
    if name in log_parameters
        all(>(0), values) ||
            throw(ArgumentError(
                "Cannot log-transform $name: nonpositive values"
            ))
        return log10.(values)
    end
    return values
end

function display_label(name, log_parameters)
    name in log_parameters ?
    "log10($(name))" : string(name)
end


"""
    plot_parameter_pairs(data; parameters, log_parameters)

Lower triangle: pairwise parameter scatterplots.
Diagonal: univariate marginal KDEs.
Upper triangle: Spearman rank correlations.
"""
function plot_parameter_pairs(
    data;
    parameters=collect(keys(data.parameters)),
    log_parameters=Symbol[],
)
    names = collect(parameters)
    n = length(names)

    fig = Figure(size=(max(800, 170*n), max(800, 170*n)))

    for i in 1:n, j in 1:n
        xname = names[j]
        yname = names[i]

        x = display_values(
            data.parameters[xname], xname, log_parameters
        )
        y = display_values(
            data.parameters[yname], yname, log_parameters
        )

        ax = Axis(fig[i, j])

        if i == j
            if length(unique(x)) >= 2
                density!(
                    ax, x;
                    color=(:steelblue, 0.35),
                    strokecolor=:steelblue,
                    strokewidth=1.5,
                )
            end

        elseif i > j
            scatter!(
                ax, x, y;
                markersize=3,
                color=(:steelblue, 0.25),
            )

        else
            r = if length(unique(x)) < 2 ||
                length(unique(y)) < 2
                NaN
            else
                corspearman(x, y)
            end

            text!(
                ax, 0.5, 0.5;
                text=isnan(r) ? "NA" : "ρ = $(round(r; digits=2))",
                align=(:center, :center),
                space=:relative,
                fontsize=16,
            )
            hidedecorations!(ax)
            hidespines!(ax)
        end

        if i != n
            hidexdecorations!(ax)
        else
            ax.xlabel = display_label(xname, log_parameters)
        end

        if j != 1
            hideydecorations!(ax)
        elseif i != j
            ax.ylabel = display_label(yname, log_parameters)
        end
    end

    return fig
end


"""
    plot_parameter_outcomes(data; parameters, outcomes,
                            log_parameters, log_outcomes)

Rows correspond to outcomes; columns to parameters.
Each panel contains a scatterplot and Spearman correlation.
"""
function plot_parameter_outcomes(
    data;
    parameters=collect(keys(data.parameters)),
    outcomes=collect(keys(data.outcomes)),
    log_parameters=Symbol[],
    log_outcomes=Symbol[],
)
    pnames = collect(parameters)
    onames = collect(outcomes)

    np = length(pnames)
    no = length(onames)

    fig = Figure(
        size=(max(900, 190*np), max(550, 210*no))
    )

    for (i, oname) in enumerate(onames)
        for (j, pname) in enumerate(pnames)

            xraw = data.parameters[pname]
            yraw = data.outcomes[oname]

            x = display_values(xraw, pname, log_parameters)
            y = display_values(yraw, oname, log_outcomes)

            ax = Axis(
                fig[i, j];
                xlabel=i == no ?
                       display_label(pname, log_parameters) : "",
                ylabel=j == 1 ?
                       display_label(oname, log_outcomes) : "",
            )

            scatter!(
                ax, x, y;
                markersize=4,
                color=(:steelblue, 0.30),
            )

            r = if length(unique(xraw)) < 2 ||
                length(unique(yraw)) < 2
                NaN
            else
                corspearman(xraw, yraw)
            end

            text!(
                ax, 0.05, 0.95;
                text=isnan(r) ? "ρ = NA" :
                     "ρ = $(round(r; digits=2))",
                align=(:left, :top),
                space=:relative,
                fontsize=14,
            )

            if i != no
                hidexdecorations!(ax)
            end

            if j != 1
                hideydecorations!(ax)
            end
        end
    end

    return fig
end



"""
    plot_prcc_heatmap(result)

Plot a parameter × outcome matrix of partial rank
correlation coefficients.
"""
function plot_prcc_heatmap(result)

    np = length(result.parameter_names)
    no = length(result.outcome_names)

    fig = Figure(
        size=(max(1000, 85*np), max(350, 110*no + 130))
    )

    ax = Axis(
        fig[1, 1];
        xticks=(1:np, string.(result.parameter_names)),
        yticks=(1:no, string.(result.outcome_names)),
        xticklabelrotation=pi/4,
        xlabel="Parameter",
        ylabel="Outcome",
    )

    hm = heatmap!(
        ax,
        1:np,
        1:no,
        permutedims(result.prcc);
        colormap=:RdBu,
        colorrange=(-1, 1),
    )

    for i in 1:no, j in 1:np
        value = result.prcc[i, j]

        text!(
            ax,
            j,
            i;
            text=string(round(value; digits=2)),
            align=(:center, :center),
            fontsize=12,
            color=abs(value) > 0.6 ? :white : :black,
        )
    end

    Colorbar(
        fig[1, 2],
        hm;
        label="Partial rank correlation",
    )

    return fig
end

