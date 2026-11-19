tinit = 1800.0
tfinal = 2024.0
tspan = (tinit, tfinal)
times = tinit:1.0:tfinal


# contact = default_contact_matrix()
params = make_parameters(
        CONTACT;
        beta = 0.75,
        progression_child = 3.,
        progression_5_14 = 0.1,
        progression_15_64 = 0.25,
        progression_65_plus = 0.5,
    )
population = get_population(EQUILIBRIUM_YEAR)    
u0 = seeded_initial_state(population)
sol = simulate(
    params;
    tspan = tspan,
    u0 = u0,
    saveat = times,
    demography = :equilibrium,
)

_, incidence_by_age = raw_annual_incidence_by_age(sol)

age_breaks = [0, 15, 25, 35, 45, 55, 65, 96]

age_group_incidence = Vector{Float64}(undef, length(age_breaks) - 1)
age_group_population = similar(age_group_incidence)

for idx in 1:(length(age_breaks) - 1)
    pop = 0.0
    incidence = 0.0
    for a in age_breaks[idx]:(age_breaks[idx+1] - 1)
        incidence += incidence_by_age[end, a+1]
        pop += population[a+1]
    end
    age_group_incidence[idx] = incidence / pop * 1e5
    age_group_population[idx] = pop
end