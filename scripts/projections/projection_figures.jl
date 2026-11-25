# Plot-only companion for calibrated_demographic_projections_with_figures.jl.
# Can be run separately from the repository root after projections complete:
#   include("scripts/projections/projection_figures.jl")
#   plot_calibrated_projection_figures("results/projections/calibrated")
using CairoMakie
using CSV, DataFrames, Statistics

function plot_calibrated_projection_figures(dir::AbstractString; use_posterior::Bool=true)
    figdir = joinpath(dir, "figures")
    mkpath(figdir)
    readfile(name) = CSV.read(joinpath(dir, name), DataFrame)
    present(name) = isfile(joinpath(dir, name))
    scenarios = (("fixed", "Fixed", :gray40),
        ("growth_only", "Population growth", :steelblue),
        ("projected", "Full demography", :darkorange))
    effects = (("population_growth", "Population growth", :steelblue),
        ("demographic_composition", "Composition/turnover", :darkorange),
        ("combined_demography", "Combined", :black))
    outcomes = (("incidence", "Incident TB cases"),
        ("notifications", "TB notifications"),
        ("mortality", "TB deaths"),
        ("incidence_rate", "Incidence / 100,000"),
        ("mortality_rate", "Mortality / 100,000"))
    bands = ["0-4", "5-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    saveboth(name, fig) = begin
        save(joinpath(figdir, name * ".png"), fig; px_per_unit=2)
        save(joinpath(figdir, name * ".pdf"), fig)
    end

    if use_posterior && present("annual_intervals.csv")
        dat = readfile("annual_intervals.csv")
        fig = Figure(size=(1250, 1100))
        for (k, (variable, ylabel)) in enumerate(outcomes)
            ax = Axis(fig[cld(k, 2), mod1(k, 2)]; title=ylabel,
                xlabel="Year", ylabel=ylabel)
            for (scenario, label, colour) in scenarios
                df = sort(filter(r -> r.scenario == scenario &&
                    r.statistic == variable, dat), :year)
                isempty(df) && continue
                band!(ax, df.year, df.lower, df.upper;
                    color=(colour, 0.14))
                lines!(ax, df.year, df.median; color=colour,
                    label=label, linewidth=2.5)
            end
            axislegend(ax; position=:lt, framevisible=false)
        end
        saveboth("national_posterior_projections", fig)
    elseif present("map_annual.csv")
        dat = readfile("map_annual.csv")
        fig = Figure(size=(1250, 1000))
        for (k, (variable, ylabel)) in enumerate(outcomes)
            ax = Axis(fig[cld(k, 2), mod1(k, 2)]; title=ylabel,
                xlabel="Year", ylabel=ylabel)
            for (scenario, label, colour) in scenarios
                df = sort(filter(r -> r.scenario == scenario, dat), :year)
                lines!(ax, df.year, df[!, Symbol(variable)];
                    color=colour, label=label, linewidth=2.5)
            end
            axislegend(ax; position=:lt, framevisible=false)
        end
        saveboth("national_map_projections", fig)
    end

    if (use_posterior && present("effects_intervals.csv")) || present("map_effects.csv")
        posterior = use_posterior && present("effects_intervals.csv")
        dat = readfile(posterior ? "effects_intervals.csv" : "map_effects.csv")
        vars = (("incidence", "Incidence (counts)"),
            ("notifications", "Notifications (counts)"),
            ("mortality", "Mortality (counts)"),
            ("mortality_rate", "Mortality rate"))
        fig = Figure(size=(1250, 850))
        for (k, (variable, title)) in enumerate(vars)
            ax = Axis(fig[cld(k, 2), mod1(k, 2)]; title=title,
                xlabel="Year", ylabel="Percentage change (%)")
            hlines!(ax, [0]; color=:gray75, linestyle=:dash)
            for (effect, label, colour) in effects
                df = sort(filter(r -> r.effect == effect &&
                    r.variable == variable &&
                    (!posterior || r.statistic == "percentage_change"),
                    dat), :year)
                isempty(df) && continue
                if posterior
                    band!(ax, df.year, df.lower, df.upper;
                        color=(colour, 0.13))
                    values = df.median
                else
                    values = df.percentage_change
                end
                lines!(ax, df.year, values; color=colour,
                    label=label, linewidth=2.5)
            end
            axislegend(ax; position=:lt, framevisible=false)
        end
        saveboth("demographic_effects", fig)
    end

    if (use_posterior && present("cumulative_effects_intervals.csv")) ||
        present("map_cumulative_effects.csv")
        posterior = use_posterior && present("cumulative_effects_intervals.csv")
        dat = readfile(posterior ? "cumulative_effects_intervals.csv" :
                       "map_cumulative_effects.csv")
        fig = Figure(size=(1200, 430))
        for (j, (variable, title)) in enumerate(outcomes[1:3])
            ax = Axis(fig[1, j]; title=title,
                ylabel="Additional cases/deaths, 2025–2050",
                xticks=(1:3, ["Growth", "Composition", "Combined"]),
                xticklabelrotation=pi/5)
            for (i, (effect, _, colour)) in enumerate(effects)
                df = filter(r -> r.effect == effect &&
                    r.variable == variable &&
                    (!posterior || r.statistic == "difference"), dat)
                isempty(df) && continue
                d = df[end, :]
                y = posterior ? d.median : d.difference
                barplot!(ax, [i], [y]; color=colour, width=0.65)
                if posterior
                    rangebars!(ax, [i], [d.lower], [d.upper];
                        color=:black, linewidth=2)
                end
            end
        end
        saveboth("cumulative_additional_burden", fig)
    end

    if (use_posterior && present("age_band_cumulative_effects_intervals.csv")) ||
        present("age_band_cumulative_effects_draws.csv")
        posterior = use_posterior && present("age_band_cumulative_effects_intervals.csv")
        dat = readfile(posterior ?
                       "age_band_cumulative_effects_intervals.csv" :
                       "age_band_cumulative_effects_draws.csv")
        if !posterior
            dat = filter(:parameter_draw => ==(0), dat)
        end
        fig = Figure(size=(1400, 800))
        for (k, (variable, title)) in enumerate(outcomes[1:3])
            ax = Axis(fig[cld(k, 2), mod1(k, 2)];
                title="Additional cumulative " * title * " by age",
                ylabel="Additional events, 2025–2050",
                xticks=(1:8, bands), xticklabelrotation=pi/5)
            for (i, (effect, label, colour)) in enumerate(effects)
                subset = filter(r -> r.effect == effect &&
                    r.variable == variable &&
                    (!posterior || r.statistic == "difference"), dat)
                for (b, age_group) in enumerate(bands)
                    df = filter(:age_group => ==(age_group), subset)
                    isempty(df) && continue
                    d = df[1, :]
                    x = b + 0.23*(i-2)
                    y = posterior ? d.median : d.difference
                    barplot!(ax, [x], [y]; color=colour,
                        width=0.21, label=(b == 1 ? label : nothing))
                    if posterior
                        rangebars!(ax, [x], [d.lower], [d.upper];
                            color=:black, linewidth=1.2)
                    end
                end
            end
            axislegend(ax; position=:lt, framevisible=false)
        end
        saveboth("cumulative_age_contributions", fig)
    end

    if (use_posterior && present("age_band_effects_intervals.csv")) ||
        present("age_band_effects_draws.csv")
        posterior = use_posterior && present("age_band_effects_intervals.csv")
        dat = readfile(posterior ?
                       "age_band_effects_intervals.csv" : "age_band_effects_draws.csv")
        if !posterior
            dat = filter(:parameter_draw => ==(0), dat)
        end
        fig = Figure(size=(1450, 810))
        for (k, (variable, title)) in enumerate(outcomes[1:3])
            ax = Axis(fig[cld(k, 2), mod1(k, 2)];
                title="2050 " * title * ": additional events by age",
                ylabel="Events per year", xticks=(1:8, bands),
                xticklabelrotation=pi/5)
            for (i, (effect, label, colour)) in enumerate(effects)
                subset = filter(r -> r.year == 2050 &&
                    r.effect == effect && r.variable == variable &&
                    (!posterior || r.statistic == "difference"), dat)
                for (b, age_group) in enumerate(bands)
                    df = filter(:age_group => ==(age_group), subset)
                    isempty(df) && continue
                    d = df[1, :]
                    x = b + 0.23*(i-2)
                    y = posterior ? d.median : d.difference
                    barplot!(ax, [x], [y]; color=colour,
                        width=0.21, label=(b == 1 ? label : nothing))
                    if posterior
                        rangebars!(ax, [x], [d.lower], [d.upper];
                            color=:black, linewidth=1.2)
                    end
                end
            end
            axislegend(ax; position=:lt, framevisible=false)
        end
        saveboth("age_effects_2050", fig)
    end

    println("Figures saved to: ", figdir)
    return figdir
end
