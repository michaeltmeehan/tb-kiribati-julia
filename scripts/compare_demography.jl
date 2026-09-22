using TBKiribatiJulia


# Simulation settings ---------------------------------------------------------

burnin_start = 1800.0
projection_start = 2024.0
projection_end = 2100.0

burnin_times = burnin_start:1.0:projection_start
projection_times = projection_start:1.0:projection_end


# Model parameters ------------------------------------------------------------

params = make_parameters(
    CONTACT;
    beta = 0.75,
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

function median_age(population_by_age)
    total = sum(population_by_age)
    target = total / 2

    cumulative = 0.0

    for (i, n) in enumerate(population_by_age)
        cumulative += n

        if cumulative >= target
            return i - 1
        end
    end

    return NAGE - 1
end


function demographic_summary(summary)
    ntime = length(summary.years)

    median_ages = Vector{Float64}(undef, ntime)
    prop_under_15 = Vector{Float64}(undef, ntime)
    prop_65_plus = Vector{Float64}(undef, ntime)

    for i in 1:ntime
        pop = @view summary.population_by_age[i, :]
        total = sum(pop)

        median_ages[i] = median_age(pop)
        prop_under_15[i] = sum(@view pop[1:15]) / total
        prop_65_plus[i] = sum(@view pop[66:end]) / total
    end

    return (
        years = summary.years,
        population = summary.population,
        median_age = median_ages,
        prop_under_15 = prop_under_15,
        prop_65_plus = prop_65_plus,
    )
end


equilibrium_demography = demographic_summary(equilibrium_summary)
dynamic_demography = demographic_summary(dynamic_summary)