using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using ADTypes: AutoFiniteDiff
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Rosenbrock23, Vern7
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
    return data, p, prob
end

function broad_totals(state)
    epi = reshape(state, NSTATE, NAGE)
    return (
        total = sum(epi[1:NEPI, :]),
        age0_4 = sum(@view epi[1:NEPI, 1:5]),
        age5_14 = sum(@view epi[1:NEPI, 6:15]),
        age15_64 = sum(@view epi[1:NEPI, 16:65]),
        age65p = sum(@view epi[1:NEPI, 66:96]),
        cumulative_infections = sum(epi[CumInfectionsOther:CumInfectionsContained, :]),
        cumulative_progression = sum(epi[CumProgressionToActiveTB, :]),
        cumulative_treatment_initiation = sum(epi[CumTreatmentInitiation, :]),
        cumulative_treatment_completion = sum(epi[CumTreatmentCompletion, :]),
        cumulative_relapse = sum(epi[CumRelapseTB, :]),
    )
end

function rel_diff(a, b)
    return abs(a - b) / max(abs(b), eps(Float64))
end

function main()
    data, p, prob = build_problem()
    baseline = solve(prob, Rosenbrock23(autodiff = AutoFiniteDiff()); saveat = 0.25, reltol = 1e-8, abstol = 1e-10)
    optimized = solve(prob, Vern7(); saveat = 0.25, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)

    println("baseline stats = ", baseline.stats)
    println("optimized stats = ", optimized.stats)

    base_final = broad_totals(baseline.u[end])
    opt_final = broad_totals(optimized.u[end])

    println("solver-vs-baseline output deltas")
    for key in keys(base_final)
        b = getproperty(base_final, key)
        o = getproperty(opt_final, key)
        println("  ", key, ": abs = ", abs(o - b), ", rel = ", rel_diff(o, b))
    end

    println("WPP demographic discrepancy by year")
    demo = solve(ODEProblem(tb_rhs!, initial_state(data.population_2025), (2025.0, 2030.0),
        make_parameters(default_contact_matrix(); demography = data.schedule, ageing_enabled = true));
        saveat = 1.0, reltol = 1e-8, abstol = 1e-10, tstops = TBKiribatiJulia.demographic_tstops(data.schedule))

    largest = (year = 0, age = 0, julia = 0.0, wpp = 0.0, abs = -Inf, rel = -Inf)
    for (i, year) in enumerate(2025:2030)
        state = reshape(demo.u[i], NSTATE, NAGE)
        sim = sum(state[1:NEPI, :], dims = 1)[:]
        ref = data.population[:, i]
        abs_total = abs(sum(sim) - sum(ref))
        rel_total = rel_diff(sum(sim), sum(ref))
        println("  year ", year, ": julia_total = ", sum(sim), ", wpp_total = ", sum(ref), ", abs = ", abs_total, ", rel = ", rel_total)

        for age in 1:NAGE
            abs_err = abs(sim[age] - ref[age])
            rel_err = rel_diff(sim[age], ref[age])
            if abs_err > largest.abs
                largest = (year = year, age = age, julia = sim[age], wpp = ref[age], abs = abs_err, rel = rel_err)
            end
        end
    end

    println("largest age-specific discrepancy")
    println("  year = ", largest.year)
    println("  age = ", TBKiribatiJulia.WPP_AGE_LABELS[largest.age])
    println("  julia = ", largest.julia)
    println("  wpp = ", largest.wpp)
    println("  abs = ", largest.abs)
    println("  rel = ", largest.rel)
end

main()
