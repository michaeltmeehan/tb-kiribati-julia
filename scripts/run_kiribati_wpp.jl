using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
using TBKiribatiJulia

data = load_kiribati_wpp_data()
u0 = initial_state(data.population_2025)
p = make_parameters(default_contact_matrix();
    beta = 1e-3,
    containment_child = 4.4,
    containment_adult = 2.0,
    clearance_rate = 0.02,
    breakdown_rate = 0.1,
    prop_infectious = 0.5,
    progression_child = 2.4,
    progression_5_14 = 2.0,
    progression_15_64 = 0.1,
    progression_65_plus = 2.4,
    clinical_progression_rate = 1.0,
    clinical_regression_rate = 1.0,
    infectiousness_gain_rate = 1.0,
    infectiousness_loss_rate = 1.0,
    self_recovery_rate = 0.4,
    detection_rate = 1.0,
    rel_detection_subclin = 0.0,
    tx_period = 0.5,
    tx_success_prop = 0.8,
    pct_neg_tx_death = 0.4,
    disease_mortality_clin_lowinf = 0.025,
    disease_mortality_clin_inf = 0.4,
    demography = data.schedule,
    ageing_enabled = true)

prob = ODEProblem(tb_rhs!, u0, (2025.0, 2030.0), p)

warm = solve(prob, Vern7(); saveat = 0.25, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)
timed = @elapsed sol = solve(prob, Vern7(); saveat = 0.25, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)

initial_state = reshape(sol.u[1], NSTATE, NAGE)
final_state = reshape(sol.u[end], NSTATE, NAGE)

age_totals_initial = sum(initial_state[1:NEPI, :], dims = 1)[:]
age_totals_final = sum(final_state[1:NEPI, :], dims = 1)[:]

function broad_total(age_totals, first_age, last_age)
    return sum(@view age_totals[first_age:last_age])
end

println("retcode = ", sol.retcode)
println("warm-up retcode = ", warm.retcode)
println("runtime after compilation/warm-up = ", timed)
println("solver stats = ", sol.stats)
println("initial total population = ", sum(age_totals_initial))
println("final total population = ", sum(age_totals_final))
println("final population by broad age group =")
println("  0-4   = ", broad_total(age_totals_final, 1, 5))
println("  5-14  = ", broad_total(age_totals_final, 6, 15))
println("  15-64 = ", broad_total(age_totals_final, 16, 65))
println("  65+   = ", broad_total(age_totals_final, 66, 96))
println("cumulative infections = ", sum(final_state[CumInfectionsOther:CumInfectionsContained, :]))
println("cumulative progression to active TB = ", sum(final_state[CumProgressionToActiveTB, :]))
println("cumulative treatment initiation = ", sum(final_state[CumTreatmentInitiation, :]))
println("cumulative treatment completion = ", sum(final_state[CumTreatmentCompletion, :]))
println("cumulative relapse = ", sum(final_state[CumRelapseTB, :]))

du = similar(u0)
fill!(du, 0.0)
tb_rhs!(du, u0, p, 2025.0)
tb_rhs!(du, u0, p, 2025.0)
println("rhs allocations per call = ", @allocated tb_rhs!(du, u0, p, 2025.0))
