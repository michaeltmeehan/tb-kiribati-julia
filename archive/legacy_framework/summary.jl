function simulation_summary(sol)
    years, incidence_by_age = raw_annual_incidence_by_age(sol)

    incidence = vec(sum(incidence_by_age; dims = 2))

    _, deaths_by_age = raw_annual_deaths_by_age(sol)

    deaths = vec(sum(deaths_by_age; dims = 2))

    population_by_age = Matrix{Float64}(undef, length(years), NAGE)

    for (i, t) in enumerate(years)
        u = sol(t, continuity = :left)
        population_by_age[i, :] .= get_age_distribution(u)
    end

    population = vec(sum(population_by_age; dims = 2))

    incidence_per_100k = 1e5 .* incidence ./ population

    deaths_per_100k = 1e5 .* deaths ./ population

    return (
        years = years,
        population = population,
        population_by_age = population_by_age,
        incidence = incidence,
        incidence_by_age = incidence_by_age,
        incidence_per_100k = incidence_per_100k,
        deaths = deaths,
        deaths_by_age = deaths_by_age,
        deaths_per_100k = deaths_per_100k
    )
end