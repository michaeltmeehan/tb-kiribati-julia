using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using ADTypes: AutoFiniteDiff
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Rosenbrock23, Tsit5, Vern7
using TBKiribatiJulia

function build_problem()
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
    return prob
end

function broad_difference(a, b)
    return maximum(abs.(a .- b))
end

function run_case(label, prob, alg; kwargs...)
    t = @elapsed sol = solve(prob, alg; kwargs...)
    println(label)
    println("  runtime (s) = ", t)
    println("  stats = ", sol.stats)
    println("  retcode = ", sol.retcode)
    return sol, t
end

function main()
    prob = build_problem()
    baseline = solve(prob, Rosenbrock23(autodiff = AutoFiniteDiff()); saveat = 0.25, reltol = 1e-8, abstol = 1e-10)
    println("baseline stats = ", baseline.stats)

    sol_tsit, _ = run_case("Tsit5", prob, Tsit5(); saveat = 0.25, reltol = 1e-8, abstol = 1e-10)
    println("  maxabs final diff vs baseline = ", broad_difference(sol_tsit.u[end], baseline.u[end]))

    sol_tsit_tstop, _ = run_case("Tsit5 + tstops", prob, Tsit5(); saveat = 0.25, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)
    println("  maxabs final diff vs baseline = ", broad_difference(sol_tsit_tstop.u[end], baseline.u[end]))

    sol_vern, _ = run_case("Vern7 + tstops", prob, Vern7(); saveat = 0.25, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)
    println("  maxabs final diff vs baseline = ", broad_difference(sol_vern.u[end], baseline.u[end]))

    sol_ros, _ = run_case("Rosenbrock23", prob, Rosenbrock23(autodiff = AutoFiniteDiff()); saveat = 0.25, reltol = 1e-8, abstol = 1e-10)
    println("  maxabs final diff vs baseline = ", broad_difference(sol_ros.u[end], baseline.u[end]))
end

main()
