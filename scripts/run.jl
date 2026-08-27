
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
import DiffEqCallbacks as CB
using DelimitedFiles

tinit = 2025.0
tfinal = 2100.0
tspan = (tinit, tfinal)
times = tinit:1.0:tfinal


# contact = default_contact_matrix()
params = make_parameters(CONTACT)
population = get_population(STATIC_YEAR)    
u0 = initial_state(population)
prob = ODEProblem(tb_rhs!, u0, tspan, params)
cb = CB.PresetTimeCallback(times, apply_demography!)
sol = solve(prob, Vern7(), callback = cb; saveat = times)

# Check age distribution time series
age_distribution = [get_age_distribution(x) for x in sol.u]