
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
import DiffEqCallbacks as CB

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
population = get_population(STATIC_YEAR)    
u0 = TBKiribatiJulia.initial_state(population)
prob = ODEProblem(tb_rhs!, u0, tspan, params)

function annual_update!(integrator)
    # Apply demography
    if integrator.t <= 2100. # STATIC_YEAR
        _apply_static_demography!(integrator.u)
    else
        _apply_demography!(integrator.u, integrator.t - 1.0)
    end
    # Reset cumulative counters
    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        for c in (NEPI + 1):NSTATE
            integrator.u[base + c] = 0.0
        end
    end
end

cb = CB.PresetTimeCallback(times, annual_update!)
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