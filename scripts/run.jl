
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
import DiffEqCallbacks as CB

tinit = 1800.0
tfinal = 2100.0
tspan = (tinit, tfinal)
times = tinit:1.0:tfinal


# contact = default_contact_matrix()
params = make_parameters(CONTACT)
params.beta = 1e-3
population = get_population(STATIC_YEAR)    
u0 = initial_state(population)
prob = ODEProblem(tb_rhs!, u0, tspan, params)

function annual_update!(integrator)
    # Apply demography
    # if integrator.t <= STATIC_YEAR
    #     _apply_static_demography!(integrator.u)
    # else
    #     _apply_demography!(integrator.u, integrator.t - 1.0)
    # end
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

# Check age distribution time series
age_distribution = [get_age_distribution(x) for x in sol.u]