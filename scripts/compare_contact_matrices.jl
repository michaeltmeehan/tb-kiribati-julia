using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Statistics: mean
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
using TBKiribatiJulia

const SAVEAT = 0.25
const TSPAN = (2025.0, 2030.0)

function build_problem(contact_matrix)
    data = load_kiribati_wpp_data()
    u0 = initial_state(data.population_2025)
    p = make_parameters(contact_matrix;
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
    return u0, p, ODEProblem(tb_rhs!, u0, TSPAN, p)
end

function solve_case(contact_matrix)
    u0, p, prob = build_problem(contact_matrix)
    warm = solve(prob, Vern7(); saveat = SAVEAT, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)
    runtime = @elapsed sol = solve(prob, Vern7(); saveat = SAVEAT, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)
    return (; u0, p, prob, warm, sol, runtime)
end

function summarize_matrix(name, matrix)
    println(name, " contact matrix")
    println("  dimensions = ", size(matrix))
    println("  minimum value = ", minimum(matrix))
    println("  maximum value = ", maximum(matrix))
    println("  mean value = ", mean(matrix))
    println("  row sums (selected ages 0,5,10,15,30,50,70,90,95) = ",
        join(round.(sum(matrix, dims = 2)[:][[1, 6, 11, 16, 31, 51, 71, 91, 96]], digits = 6), ", "))
    println("  symmetry max abs diff = ", maximum(abs.(matrix .- transpose(matrix))))
end

function summarize_solution(name, result)
    state = reshape(result.sol.u[end], NSTATE, NAGE)
    population = sum(state[1:NEPI, :])
    cum_infections = sum(state[CumInfectionsOther:CumInfectionsContained, :])
    cum_progression = sum(state[CumProgressionToActiveTB, :])
    cum_tx = sum(state[CumTreatmentInitiation, :])
    println(name, " run")
    println("  solver success = ", occursin("Success", string(result.sol.retcode)))
    println("  runtime = ", result.runtime)
    println("  final population = ", population)
    println("  cumulative infections = ", cum_infections)
    println("  cumulative progression to active TB = ", cum_progression)
    println("  cumulative treatment initiation = ", cum_tx)
    println("  force of infection at selected ages =")
    λ = zeros(Float64, NAGE)
    compute_force_of_infection!(λ, result.u0, result.p)
    for age in (0, 5, 10, 15, 20, 30, 40, 50, 65, 80, 95)
        println("    age ", age, " => ", λ[age + 1])
    end
end

function main()
    synthetic = default_contact_matrix()
    empirical = load_solomon_islands_contact_matrix()

    summarize_matrix("synthetic", synthetic)
    summarize_matrix("empirical", empirical)

    synth = solve_case(synthetic)
    empir = solve_case(empirical)

    summarize_solution("synthetic", synth)
    summarize_solution("empirical", empir)
end

main()
