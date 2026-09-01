using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
import DiffEqCallbacks as CB

tinit = 1800.0
tfinal = 2025.0
tspan = (tinit, tfinal)
times = tinit:1.0:tfinal


# contact = default_contact_matrix()
params = make_parameters(CONTACT)
population = get_population(STATIC_YEAR)    
u0 = initial_state(population)
prob = ODEProblem(tb_rhs!, u0, tspan, params)
cb = CB.PresetTimeCallback(times, apply_demography!)
sol = solve(prob, Vern7(), callback = cb; saveat = times)


betas = collect(0.1:0.1:1.0)

annual_incidence = Matrix{Float64}(undef, length(times) - 1, length(betas))
no_demography = Matrix{Float64}(undef, length(times) - 1, length(betas))

for (i, beta) in enumerate(betas)
    params.beta = beta
    prob = ODEProblem(tb_rhs!, u0, tspan, params)
    sol = solve(prob, Vern7(), callback = cb; saveat = times)
    _, incidence_by_age = raw_annual_incidence_by_age(sol)
    population = [sum(sol(t)) for t in times[2:end]]
    annual_incidence[:, i] .= sum(incidence_by_age, dims=2) .* 1e5 ./ population

    sol_no_demography = solve(prob, Vern7(), saveat = times)
    _, incidence_by_age = raw_annual_incidence_by_age(sol_no_demography)
    no_demography[:, i] .= sum(incidence_by_age, dims=2) ./ population .* 1e5
end

# β = 0.593 gives 335 / 1e5 incidence

population = get_population(STATIC_YEAR) 
beta = 0.583
params.beta = beta
prob = ODEProblem(tb_rhs!, u0, tspan, params)
sol = solve(prob, Vern7(), callback = cb; saveat = times)
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

