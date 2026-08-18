using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using TBKiribatiJulia

schedule = synthetic_demographic_schedule()
sol = simulate_demographic_demo(schedule = schedule)

initial_solution_state = reshape(sol.u[1], NSTATE, NAGE)
final_state = reshape(sol.u[end], NSTATE, NAGE)

initial_total = sum(initial_solution_state[1:NEPI, :])
final_total = sum(final_state[1:NEPI, :])
initial_95plus = sum(initial_solution_state[1:NEPI, 96])
final_95plus = sum(final_state[1:NEPI, 96])

println("retcode = ", sol.retcode)
println("initial total population = ", initial_total)
println("final total population = ", final_total)
println("initial population age 95+ = ", initial_95plus)
println("final population age 95+ = ", final_95plus)
println("cumulative infections = ", sum(final_state[CumInfectionsOther:CumInfectionsContained, :]))
println("cumulative progression to active TB = ", sum(final_state[CumProgressionToActiveTB, :]))
println("cumulative treatment initiation = ", sum(final_state[CumTreatmentInitiation, :]))

params = make_demographic_parameters(default_contact_matrix(), schedule)
u0 = TBKiribatiJulia.initial_state(default_population())
du = similar(u0)

function measure_allocations!(du, u0, params)
    for trial in 1:5
        fill!(du, 0.0)
        tb_rhs!(du, u0, params, 2025.0)
    end
    rhs_alloc = @allocated tb_rhs!(du, u0, params, 2025.0)
    dem_alloc = @allocated apply_demography!(du, u0, params, 2025.0)
    return rhs_alloc, dem_alloc
end

rhs_alloc, dem_alloc = measure_allocations!(du, u0, params)

println("rhs allocations per call = ", rhs_alloc)
println("demography allocations per call = ", dem_alloc)
