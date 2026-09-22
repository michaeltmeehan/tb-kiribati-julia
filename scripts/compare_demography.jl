using TBKiribatiJulia


# Simulation settings ---------------------------------------------------------

burnin_start = 1800.0
projection_start = 2025.0
projection_end = 2100.0

burnin_times = burnin_start:1.0:projection_start
projection_times = projection_start:1.0:projection_end


# Model parameters ------------------------------------------------------------

params = make_parameters(
    CONTACT;
    beta = 0.8,
    progression_child = 3.0,
    progression_5_14 = 0.1,
    progression_15_64 = 0.25,
    progression_65_plus = 0.5,
    infectiousness_weights = (0.2, 0.5, 0.4, 1.0),

    # TB-related mortality
    pct_neg_tx_death = 0.4,
    disease_mortality_clin_lowinf = 0.025,
    disease_mortality_clin_inf = 0.4,
)


# Common burn-in --------------------------------------------------------------

population0 = get_population(EQUILIBRIUM_YEAR)

u0 = seeded_initial_state(population0)

burnin_sol = simulate(
    params;
    tspan = (burnin_start, projection_start),
    u0 = u0,
    saveat = burnin_times,
    demography = :equilibrium,
)

# Both projection scenarios start from exactly the same epidemiological state.
projection_u0 = copy(burnin_sol.u[end])


# Equilibrium-demography projection -------------------------------------------

equilibrium_sol = simulate(
    params;
    tspan = (projection_start, projection_end),
    u0 = copy(projection_u0),
    saveat = projection_times,
    demography = :equilibrium,
)


# Dynamic-demography projection -----------------------------------------------

dynamic_sol = simulate(
    params;
    tspan = (projection_start, projection_end),
    u0 = copy(projection_u0),
    saveat = projection_times,
    demography = :dynamic,
)


# Epidemiological summaries ---------------------------------------------------

equilibrium_summary = simulation_summary(equilibrium_sol)
dynamic_summary = simulation_summary(dynamic_sol)


# Demographic summaries -------------------------------------------------------

equilibrium_demography = demographic_summary(equilibrium_summary)
dynamic_demography = demographic_summary(dynamic_summary)


using Plots

# Overall TB incidence --------------------------------------------------------

plot(
           equilibrium_summary.years,
           equilibrium_summary.incidence_per_100k;
           label = "Equilibrium demography",
           xlabel = "Year",
           ylabel = "TB incidence per 100,000",
           linewidth = 2,
           ylims = (0, 1_000),
       )

plot!(
           dynamic_summary.years,
           dynamic_summary.incidence_per_100k;
           label = "Dynamic demography",
           linewidth = 2,
       )



# Relative difference in TB incidence -----------------------------------------

incidence_relative_difference = 100 .* (
    dynamic_summary.incidence_per_100k ./
    equilibrium_summary.incidence_per_100k .- 1
)

plot(
    equilibrium_summary.years,
    incidence_relative_difference;
    label = false,
    xlabel = "Year",
    ylabel = "Difference in TB incidence (%)",
    linewidth = 2,
    legend = false,
)

hline!([0]; linestyle = :dash, label = false)


# Median age ------------------------------------------------------------------

plot(
    equilibrium_demography.years,
    equilibrium_demography.median_age;
    label = "Equilibrium demography",
    xlabel = "Year",
    ylabel = "Median age (years)",
    linewidth = 2,
    ylims = (0, 50),
)

plot!(
    dynamic_demography.years,
    dynamic_demography.median_age;
    label = "Dynamic demography",
    linewidth = 2,
)


# Proportion aged 65+ ---------------------------------------------------------

plot(
    equilibrium_demography.years,
    100 .* equilibrium_demography.prop_65_plus;
    label = "Equilibrium demography",
    xlabel = "Year",
    ylabel = "Population aged 65+ (%)",
    linewidth = 2,
    ylims = (0, 17)
)

plot!(
    dynamic_demography.years,
    100 .* dynamic_demography.prop_65_plus;
    label = "Dynamic demography",
    linewidth = 2,
)


# TB mortality ---------------------------------------------------------------

plot(
    equilibrium_summary.years,
    equilibrium_summary.deaths_per_100k;
    label = "Equilibrium demography",
    xlabel = "Year",
    ylabel = "TB deaths per 100,000",
    linewidth = 2,
    ylim = (0, 100)
)

plot!(
    dynamic_summary.years,
    dynamic_summary.deaths_per_100k;
    label = "Dynamic demography",
    linewidth = 2,
)



using DataFrames

# Summary table ---------------------------------------------------------------

summary_years = [2050, 2075, 2100]

rows = DataFrame()

    push!(rows, (
        year = 2025,
        scenario = "Equilibrium",
        incidence_per_100k = equilibrium_summary.incidence_per_100k[1],
        deaths_per_100k = equilibrium_summary.deaths_per_100k[1],
        median_age = equilibrium_demography.median_age[1],
        prop_65_plus = 100 * equilibrium_demography.prop_65_plus[1],
        prop_under_5 = 100 * equilibrium_demography.prop_under_5[1],
    ))

for year in summary_years
    i = findfirst(==(year), equilibrium_summary.years)

    # push!(rows, (
    #     year = year,
    #     scenario = "Equilibrium",
    #     incidence_per_100k = equilibrium_summary.incidence_per_100k[i],
    #     deaths_per_100k = equilibrium_summary.deaths_per_100k[i],
    #     median_age = equilibrium_demography.median_age[i],
    #     prop_65_plus = 100 * equilibrium_demography.prop_65_plus[i],
    #     prop_under_5 = 100 * equilibrium_demography.prop_under_5[i],
    # ))

    push!(rows, (
        year = year,
        scenario = "Dynamic",
        incidence_per_100k = dynamic_summary.incidence_per_100k[i],
        deaths_per_100k = dynamic_summary.deaths_per_100k[i],
        median_age = dynamic_demography.median_age[i],
        prop_65_plus = 100 * dynamic_demography.prop_65_plus[i],
        prop_under_5 = 100 * dynamic_demography.prop_under_5[i],
    ))
end

rows