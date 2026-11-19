function run_demography_comparison(
    params::TBParams;
    burnin_start::Real = 1800.0,
    projection_start::Real = 2025.0,
    projection_end::Real = 2100.0,
)
    burnin_times = burnin_start:1.0:projection_start
    projection_times = projection_start:1.0:projection_end

    population0 = get_population(EQUILIBRIUM_YEAR)
    u0 = seeded_initial_state(population0)

    burnin_sol = simulate(
        params;
        tspan = (burnin_start, projection_start),
        u0 = u0,
        saveat = burnin_times,
        demography = :equilibrium,
    )

    projection_u0 = copy(burnin_sol.u[end])

    equilibrium_sol = simulate(
        params;
        tspan = (projection_start, projection_end),
        u0 = copy(projection_u0),
        saveat = projection_times,
        demography = :equilibrium,
    )

    dynamic_sol = simulate(
        params;
        tspan = (projection_start, projection_end),
        u0 = copy(projection_u0),
        saveat = projection_times,
        demography = :dynamic,
    )

    return (
        burnin = burnin_sol,
        equilibrium = equilibrium_sol,
        dynamic = dynamic_sol,
    )
end


function comparison_summary(result)
    equilibrium = simulation_summary(result.equilibrium)
    dynamic = simulation_summary(result.dynamic)

    equilibrium.years == dynamic.years ||
        error("equilibrium and dynamic summaries use different years")

    incidence_ratio =
        dynamic.incidence_per_100k ./
        equilibrium.incidence_per_100k

    mortality_ratio =
        dynamic.deaths_per_100k ./
        equilibrium.deaths_per_100k

    return (
        years = equilibrium.years,

        equilibrium_incidence_per_100k =
            equilibrium.incidence_per_100k,
        dynamic_incidence_per_100k =
            dynamic.incidence_per_100k,

        incidence_ratio = incidence_ratio,
        incidence_relative_difference =
            100 .* (incidence_ratio .- 1),
        incidence_log_ratio =
            log.(incidence_ratio),

        equilibrium_deaths_per_100k =
            equilibrium.deaths_per_100k,
        dynamic_deaths_per_100k =
            dynamic.deaths_per_100k,

        mortality_ratio = mortality_ratio,
        mortality_relative_difference =
            100 .* (mortality_ratio .- 1),
        mortality_log_ratio =
            log.(mortality_ratio),
    )
end