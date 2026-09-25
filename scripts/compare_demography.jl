using TBKiribatiJulia
using Plots
using DataFrames


# Simulation settings ---------------------------------------------------------

burnin_start = 1800.0
projection_start = 2025.0
projection_end = 2100.0


# Model parameters ------------------------------------------------------------

params = make_parameters(
    CONTACT;
    beta = 1.1,
    progression_child = 0.5,
    progression_5_14 = 0.3,
    progression_15_64 = 0.15,
    progression_65_plus = 0.3,
    infectiousness_weights = (0.2, 0.5, 0.4, 1.0),

    # TB-related mortality
    pct_neg_tx_death = 0.4,
    disease_mortality_clin_lowinf = 0.025,
    disease_mortality_clin_inf = 0.4,
)


# Run paired demographic projections -----------------------------------------

result = run_demography_comparison(
    params;
    burnin_start = burnin_start,
    projection_start = projection_start,
    projection_end = projection_end,
)

burnin_sol = result.burnin
equilibrium_sol = result.equilibrium
dynamic_sol = result.dynamic


# Epidemiological summaries ---------------------------------------------------

equilibrium_summary = simulation_summary(equilibrium_sol)
dynamic_summary = simulation_summary(dynamic_sol)

comparison = comparison_summary(result)


# Demographic summaries -------------------------------------------------------

equilibrium_demography = demographic_summary(equilibrium_summary)
dynamic_demography = demographic_summary(dynamic_summary)


# Overall TB incidence --------------------------------------------------------

Plots.plot(
    equilibrium_summary.years,
    equilibrium_summary.incidence_per_100k;
    label = "Equilibrium demography",
    xlabel = "Year",
    ylabel = "TB incidence per 100,000",
    linewidth = 2,
    ylims = (0, 1_000),
)

Plots.plot!(
    dynamic_summary.years,
    dynamic_summary.incidence_per_100k;
    label = "Dynamic demography",
    linewidth = 2,
)


# Relative difference in TB incidence -----------------------------------------

incidence_relative_difference =
    comparison.incidence_relative_difference

Plots.plot(
    equilibrium_summary.years,
    incidence_relative_difference;
    label = false,
    xlabel = "Year",
    ylabel = "Difference in TB incidence (%)",
    linewidth = 2,
    legend = false,
)

Plots.hline!([0]; linestyle = :dash, label = false)


# Median age ------------------------------------------------------------------

Plots.plot(
    equilibrium_demography.years,
    equilibrium_demography.median_age;
    label = "Equilibrium demography",
    xlabel = "Year",
    ylabel = "Median age (years)",
    linewidth = 2,
    ylims = (0, 50),
)

Plots.plot!(
    dynamic_demography.years,
    dynamic_demography.median_age;
    label = "Dynamic demography",
    linewidth = 2,
)


# Proportion aged 65+ ---------------------------------------------------------

Plots.plot(
    equilibrium_demography.years,
    100 .* equilibrium_demography.prop_65_plus;
    label = "Equilibrium demography",
    xlabel = "Year",
    ylabel = "Population aged 65+ (%)",
    linewidth = 2,
    ylims = (0, 17),
)

Plots.plot!(
    dynamic_demography.years,
    100 .* dynamic_demography.prop_65_plus;
    label = "Dynamic demography",
    linewidth = 2,
)


# TB mortality ----------------------------------------------------------------

Plots.plot(
    equilibrium_summary.years,
    equilibrium_summary.deaths_per_100k;
    label = "Equilibrium demography",
    xlabel = "Year",
    ylabel = "TB deaths per 100,000",
    linewidth = 2,
    ylims = (0, 100),
)

Plots.plot!(
    dynamic_summary.years,
    dynamic_summary.deaths_per_100k;
    label = "Dynamic demography",
    linewidth = 2,
)


# Relative difference in TB mortality -----------------------------------------

mortality_relative_difference =
    comparison.mortality_relative_difference

Plots.plot(
    equilibrium_summary.years,
    mortality_relative_difference;
    label = false,
    xlabel = "Year",
    ylabel = "Difference in TB mortality (%)",
    linewidth = 2,
    legend = false,
)

Plots.hline!([0]; linestyle = :dash, label = false)


# Summary table ---------------------------------------------------------------

summary_years = [2050.0, 2075.0, projection_end]

rows = DataFrame(
    year = Float64[],
    scenario = String[],
    incidence_per_100k = Float64[],
    deaths_per_100k = Float64[],
    median_age = Float64[],
    prop_65_plus = Float64[],
    prop_under_5 = Float64[],
)

for year in summary_years
    i_eq = findfirst(==(year), equilibrium_summary.years)
    i_dyn = findfirst(==(year), dynamic_summary.years)

    push!(rows, (
        year = year,
        scenario = "Equilibrium",
        incidence_per_100k = equilibrium_summary.incidence_per_100k[i_eq],
        deaths_per_100k = equilibrium_summary.deaths_per_100k[i_eq],
        median_age = equilibrium_demography.median_age[i_eq],
        prop_65_plus = 100 * equilibrium_demography.prop_65_plus[i_eq],
        prop_under_5 = 100 * equilibrium_demography.prop_under_5[i_eq],
    ))

    push!(rows, (
        year = year,
        scenario = "Dynamic",
        incidence_per_100k = dynamic_summary.incidence_per_100k[i_dyn],
        deaths_per_100k = dynamic_summary.deaths_per_100k[i_dyn],
        median_age = dynamic_demography.median_age[i_dyn],
        prop_65_plus = 100 * dynamic_demography.prop_65_plus[i_dyn],
        prop_under_5 = 100 * dynamic_demography.prop_under_5[i_dyn],
    ))
end

rows