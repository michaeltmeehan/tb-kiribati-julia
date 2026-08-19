using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using BenchmarkTools
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
using Statistics: mean, median
using TBKiribatiJulia

function build_kiribati_wpp_problem()
    data = load_kiribati_wpp_data()
    u0 = initial_state(data.population_2025)
    p = make_parameters(load_solomon_islands_contact_matrix();
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
    return data, u0, p, prob
end

function summarize_trial(label::AbstractString, trial)
    println(label)
    println("  min runtime (s) = ", minimum(trial).time / 1e9)
    println("  median runtime (s) = ", median(trial).time / 1e9)
    println("  mean runtime (s) = ", mean(trial).time / 1e9)
    println("  allocations = ", trial.allocs)
    println("  memory (bytes) = ", trial.memory)
end

function main()
    data, u0, p, prob = build_kiribati_wpp_problem()
    warm = solve(prob, Vern7(); saveat = 0.25, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)
    println("warm-up retcode = ", warm.retcode)
    println("warm-up stats = ", warm.stats)
    println("warm-up stat fields = ", propertynames(warm.stats))
    println("solver algorithm = ", typeof(warm.alg))

    du = similar(u0)
    for t in (2025.25, 2025.999, 2026.0, 2028.5)
        tb_rhs!(du, u0, p, t)
        tb_rhs!(du, u0, p, t)
        empty_trial = @benchmark tb_rhs!($du, $u0, $p, $t) evals=1 samples=100
        println("rhs @ t=", t)
        println("  time (ns) = ", minimum(empty_trial).time)
        println("  allocs = ", empty_trial.allocs)
        println("  memory (bytes) = ", empty_trial.memory)
    end

    solve_trial = @benchmark solve($prob, Vern7(); saveat = 0.25, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0) evals=1 samples=50
    summarize_trial("full solve", solve_trial)
end

main()
