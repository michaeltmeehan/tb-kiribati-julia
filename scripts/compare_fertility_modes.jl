using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Printf
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
using TBKiribatiJulia

# :wpp is the preferred fertility mode for sex-aggregated simulations because it uses
# published WPP annual births directly. :agepi_compatible is retained for regression
# testing and reproducibility against the original R implementation.
const START_YEAR = 2025.0
const END_YEAR = 2030.0
const SAVEAT = 0.25

function build_parameters(schedule)
    return make_parameters(default_contact_matrix();
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
        demography = schedule,
        ageing_enabled = true)
end

function broad_total(age_totals, first_age::Int, last_age::Int)
    return sum(@view age_totals[first_age:last_age])
end

function total_population(state)
    return sum(reshape(state, NSTATE, NAGE)[1:NEPI, :])
end

function age_totals(state)
    return vec(sum(reshape(state, NSTATE, NAGE)[1:NEPI, :], dims = 1))
end

function published_total(pop_matrix, year::Int)
    return sum(@view pop_matrix[:, year - first(WPP_YEARS) + 1])
end

function extract_year_state(sol, year::Int)
    idx = findfirst(==(Float64(year)), sol.t)
    idx === nothing && error("Missing saved state for year $year")
    return reshape(sol.u[idx], NSTATE, NAGE)
end

function extract_final_summary(sol, published_population)
    final_state = reshape(sol.u[end], NSTATE, NAGE)
    final_age_totals = vec(sum(final_state[1:NEPI, :], dims = 1))
    final_total = sum(final_age_totals)

    epi = (
        cumulative_infections = sum(final_state[CumInfectionsOther:CumInfectionsContained, :]),
        cumulative_progression_to_active_tb = sum(final_state[CumProgressionToActiveTB, :]),
        cumulative_treatment_initiation = sum(final_state[CumTreatmentInitiation, :]),
        cumulative_treatment_completion = sum(final_state[CumTreatmentCompletion, :]),
        cumulative_relapse = sum(final_state[CumRelapseTB, :]),
    )

    return (
        total = final_total,
        age0 = final_age_totals[1],
        broad = (
            age0_4 = broad_total(final_age_totals, 1, 5),
            age5_14 = broad_total(final_age_totals, 6, 15),
            age15_64 = broad_total(final_age_totals, 16, 65),
            age65_plus = broad_total(final_age_totals, 66, 96),
        ),
        epi = epi,
        annual = [total_population(extract_year_state(sol, year)) for year in WPP_YEARS],
        published = [published_total(published_population, year) for year in WPP_YEARS],
    )
end

function run_mode(label::AbstractString, fertility_mode::Symbol, base_data)
    schedule = if fertility_mode === :agepi_compatible
        base_data.schedule
    elseif fertility_mode === :wpp
        DemographicSchedule(base_data.schedule.years, base_data.mortality, base_data.fertility, base_data.migration;
            births = base_data.births,
            fertility_mode = :wpp)
    else
        error("Unsupported fertility mode: $fertility_mode")
    end

    u0 = initial_state(base_data.population_2025)
    p = build_parameters(schedule)
    prob = ODEProblem(tb_rhs!, u0, (START_YEAR, END_YEAR), p)
    tstops = TBKiribatiJulia.demographic_tstops(schedule)

    warm = solve(prob, Vern7(); saveat = SAVEAT, reltol = 1e-8, abstol = 1e-10, tstops = tstops)
    warm_ok = occursin("Success", string(warm.retcode))
    warm_ok || error("$label warm-up failed with retcode $(warm.retcode)")

    runtime = @elapsed sol = solve(prob, Vern7(); saveat = SAVEAT, reltol = 1e-8, abstol = 1e-10, tstops = tstops)
    ok = occursin("Success", string(sol.retcode))
    ok || error("$label solve failed with retcode $(sol.retcode)")

    summary = extract_final_summary(sol, base_data.population)
    return (; label, fertility_mode, runtime, u0, sol, summary)
end

function pop_error(simulated::Real, published::Real)
    abs_err = abs(simulated - published)
    rel_err = abs_err / max(abs(published), eps(Float64))
    return abs_err, rel_err
end

function print_annual_population_comparison(agepi_summary, wpp_summary)
    println()
    println("Annual total population")
    println("Year | agepi-compatible | WPP-mode | published WPP | abs err agepi | rel err agepi | abs err WPP | rel err WPP")
    for i in eachindex(agepi_summary.annual)
        year = first(WPP_YEARS) + (i - 1)
        published = agepi_summary.published[i]
        agepi_value = agepi_summary.annual[i]
        wpp_value = wpp_summary.annual[i]
        agepi_abs, agepi_rel = pop_error(agepi_value, published)
        wpp_abs, wpp_rel = pop_error(wpp_value, published)
        @printf("%4d | %15.3f | %8.3f | %13.3f | %13.3f | %12.6f | %11.3f | %10.6f\n",
            year, agepi_value, wpp_value, published, agepi_abs, agepi_rel, wpp_abs, wpp_rel)
    end
end

function print_epi_comparison(agepi_summary, wpp_summary)
    println()
    println("Epidemiological comparison")
    println("Metric | agepi-compatible | WPP-mode | abs diff | rel diff")
    metrics = (
        ("cumulative infections", agepi_summary.epi.cumulative_infections, wpp_summary.epi.cumulative_infections),
        ("cumulative progression to active TB", agepi_summary.epi.cumulative_progression_to_active_tb, wpp_summary.epi.cumulative_progression_to_active_tb),
        ("cumulative treatment initiation", agepi_summary.epi.cumulative_treatment_initiation, wpp_summary.epi.cumulative_treatment_initiation),
        ("cumulative treatment completion", agepi_summary.epi.cumulative_treatment_completion, wpp_summary.epi.cumulative_treatment_completion),
        ("cumulative relapse", agepi_summary.epi.cumulative_relapse, wpp_summary.epi.cumulative_relapse),
    )
    for (name, agepi_value, wpp_value) in metrics
        abs_diff = abs(agepi_value - wpp_value)
        rel_diff = abs_diff / max(abs(wpp_value), eps(Float64))
        @printf("%-36s | %16.3f | %8.3f | %8.3f | %8.6f\n", name, agepi_value, wpp_value, abs_diff, rel_diff)
    end
end

function print_mode_summary(label, result)
    println()
    println(label)
    @printf("runtime (s): %.3f\n", result.runtime)
    @printf("initial population: %.3f\n", total_population(result.u0))
    @printf("2030 total population: %.3f\n", result.summary.total)
    @printf("2030 population aged 0: %.3f\n", result.summary.age0)
    println("2030 broad age-group population:")
    @printf("  0-4   = %.3f\n", result.summary.broad.age0_4)
    @printf("  5-14  = %.3f\n", result.summary.broad.age5_14)
    @printf("  15-64 = %.3f\n", result.summary.broad.age15_64)
    @printf("  65+   = %.3f\n", result.summary.broad.age65_plus)
end

function main()
    data = load_kiribati_wpp_data()
    agepi = run_mode("agepi-compatible", :agepi_compatible, data)
    wpp = run_mode("WPP", :wpp, data)

    println("Fertility mode comparison")
    println("Both runs succeeded: ", occursin("Success", string(agepi.sol.retcode)) && occursin("Success", string(wpp.sol.retcode)))

    print_mode_summary("agepi-compatible mode", agepi)
    print_mode_summary("WPP mode", wpp)
    print_annual_population_comparison(agepi.summary, wpp.summary)
    print_epi_comparison(agepi.summary, wpp.summary)

    agepi_2030_pub = last(agepi.summary.published)
    agepi_2030_abs, agepi_2030_rel = pop_error(last(agepi.summary.annual), agepi_2030_pub)
    wpp_2030_abs, wpp_2030_rel = pop_error(last(wpp.summary.annual), agepi_2030_pub)
    println()
    @printf("2030 population error vs published WPP: agepi-compatible abs=%.3f rel=%.6f | WPP-mode abs=%.3f rel=%.6f\n",
        agepi_2030_abs, agepi_2030_rel, wpp_2030_abs, wpp_2030_rel)
end

main()
