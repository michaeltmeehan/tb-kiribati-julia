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