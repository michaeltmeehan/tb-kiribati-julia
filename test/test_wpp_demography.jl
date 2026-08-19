using Test
using ADTypes: AutoFiniteDiff
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Rosenbrock23

const WPP_DATA_DIR = joinpath(@__DIR__, "..", "data", "wpp2024")
const WPP_REFERENCE_DIR = joinpath(@__DIR__, "..", "validation", "reference", "agepi_kiribati")

const JULIA_COMPARTMENTS = (
    "M.tb" => MtbNaive,
    "Contained" => Contained,
    "Cleared" => Cleared,
    "Recovered" => Recovered,
    "Incipient" => Incipient,
    "Sub.clin.lowinf" => SubClinLow,
    "Sub.clin.inf" => SubClinInf,
    "Clin.lowinf" => ClinLow,
    "Clin.inf" => ClinInf,
    "Treatment" => Treatment,
)

function _max_abs_diff(a, b)
    return maximum(abs.(a .- b))
end

function _max_rel_diff(a, b)
    denom = max.(abs.(b), eps(Float64))
    return maximum(abs.(a .- b) ./ denom)
end

function _total_diff(a, b)
    return sum(a) - sum(b)
end

function _time_col_index(years, year)
    idx = findfirst(==(year), years)
    idx === nothing && error("Missing year $year")
    return idx
end

function _state_long_table(sol)
    rows = NamedTuple{(:time, :age, :compartment, :value), Tuple{Float64, String, String, Float64}}[]
    compartment_names = String[first(pair) for pair in JULIA_COMPARTMENTS]
    for (ti, t) in enumerate(sol.t)
        state = reshape(sol.u[ti], NSTATE, NAGE)
        for (compartment_name, compartment_idx) in JULIA_COMPARTMENTS
            for age in 1:NAGE
                push!(rows, (
                    time = Float64(t),
                    age = TBKiribatiJulia.WPP_AGE_LABELS[age],
                    compartment = compartment_name,
                    value = state[compartment_idx, age],
                ))
            end
        end
    end
    return rows
end

function _read_long_csv(path)
    headers, rows = TBKiribatiJulia._read_simple_csv(path)
    idx = Dict(name => i for (i, name) in enumerate(headers))
    return headers, rows, idx
end

function _csv_value(rows, idx, rownum, col)
    return rows[rownum][idx[col]]
end

function _reference_state_map(path)
    headers, rows, idx = _read_long_csv(path)
    needed = ("time", "age_group", "compartment", "value")
    for col in needed
        haskey(idx, col) || error("Missing column $col in $path")
    end
    out = Dict{Tuple{Float64, String, String}, Float64}()
    for (r, _) in pairs(rows)
        key = (
            parse(Float64, _csv_value(rows, idx, r, "time")),
            _csv_value(rows, idx, r, "age_group"),
            _csv_value(rows, idx, r, "compartment"),
        )
        out[key] = parse(Float64, _csv_value(rows, idx, r, "value"))
    end
    return out
end

function _reference_state_series(path)
    headers, rows, idx = _read_long_csv(path)
    compartment_map = Dict(JULIA_COMPARTMENTS)
    for col in ("time", "age_group", "compartment", "value")
        haskey(idx, col) || error("Missing column $col in $path")
    end
    times = sort(unique(parse.(Float64, [_csv_value(rows, idx, r, "time") for r in eachindex(rows)])))
    time_to_matrix = Dict{Float64, Matrix{Float64}}(t => zeros(Float64, NSTATE, NAGE) for t in times)
    for (r, _) in pairs(rows)
        t = parse(Float64, _csv_value(rows, idx, r, "time"))
        age = findfirst(==(_csv_value(rows, idx, r, "age_group")), TBKiribatiJulia.WPP_AGE_LABELS)
        age === nothing && error("Unknown age group in $path")
        comp = get(compartment_map, _csv_value(rows, idx, r, "compartment"), nothing)
        comp === nothing && error("Unknown compartment in $path")
        time_to_matrix[t][comp, age] = parse(Float64, _csv_value(rows, idx, r, "value"))
    end
    states = [time_to_matrix[t] for t in times]
    return times, states
end

function _state_map_from_solution(sol)
    out = Dict{Tuple{Float64, String, String}, Float64}()
    for (ti, t) in enumerate(sol.t)
        state = reshape(sol.u[ti], NSTATE, NAGE)
        for (compartment_name, compartment_idx) in JULIA_COMPARTMENTS
            for age in 1:NAGE
                out[(Float64(t), TBKiribatiJulia.WPP_AGE_LABELS[age], compartment_name)] = state[compartment_idx, age]
            end
        end
    end
    return out
end

function _state_series_from_solution(sol)
    times = Float64.(sol.t)
    states = [reshape(u, NSTATE, NAGE) for u in sol.u]
    return times, states
end

function _flow_totals_from_state(state, p)
    λ = similar(p.tmp_foi)
    compute_force_of_infection!(λ, vec(state), p)
    totals = zeros(Float64, 6)
    @inbounds for a in 1:NAGE
        λa = λ[a]
        naive = state[MtbNaive, a]
        contained = state[Contained, a]
        cleared = state[Cleared, a]
        recovered = state[Recovered, a]
        incipient = state[Incipient, a]
        sublow = state[SubClinLow, a]
        subinf = state[SubClinInf, a]
        clinlow = state[ClinLow, a]
        clininf = state[ClinInf, a]
        tx = state[Treatment, a]

        inf_naive = λa * p.susceptibility_naive[a] * naive
        inf_contained = λa * p.susceptibility_contained[a] * contained
        inf_cleared = λa * p.susceptibility_cleared[a] * cleared
        inf_recovered = λa * p.susceptibility_recovered[a] * recovered
        to_sub_low = (1 - p.prop_infectious) * p.progression_rate[a] * incipient
        to_sub_inf = p.prop_infectious * p.progression_rate[a] * incipient
        to_tx_from_sub_low = p.rel_detection_subclin * p.detection_rate * sublow
        to_tx_from_sub_inf = p.rel_detection_subclin * p.detection_rate * subinf
        to_tx_from_clin_low = p.detection_rate * clinlow
        to_tx_from_clin_inf = p.detection_rate * clininf
        tx_to_rec = p.tx_recovery_rate * tx
        tx_to_relapse = p.tx_relapse_rate * tx

        totals[1] += inf_naive + inf_cleared + inf_recovered
        totals[2] += inf_contained
        totals[3] += to_sub_low + to_sub_inf
        totals[4] += to_tx_from_sub_low + to_tx_from_sub_inf + to_tx_from_clin_low + to_tx_from_clin_inf
        totals[5] += tx_to_rec
        totals[6] += tx_to_relapse
    end
    return totals
end

function _cumulative_flow_series(times, states, p)
    n = length(times)
    series = zeros(Float64, 6, n)
    if n == 0
        return series
    end
    prev_rates = _flow_totals_from_state(states[1], p)
    for i in 2:n
        rates = _flow_totals_from_state(states[i], p)
        dt = times[i] - times[i - 1]
        series[:, i] = series[:, i - 1] + 0.5 .* (prev_rates .+ rates) .* dt
        prev_rates = rates
    end
    return series
end

function _make_zero_epi_params(schedule)
    p = make_parameters(default_contact_matrix();
        beta = 0.0,
        containment_child = 0.0,
        containment_adult = 0.0,
        clearance_rate = 0.0,
        breakdown_rate = 0.0,
        prop_infectious = 0.0,
        progression_child = 0.0,
        progression_5_14 = 0.0,
        progression_15_64 = 0.0,
        progression_65_plus = 0.0,
        clinical_progression_rate = 0.0,
        clinical_regression_rate = 0.0,
        infectiousness_gain_rate = 0.0,
        infectiousness_loss_rate = 0.0,
        self_recovery_rate = 0.0,
        detection_rate = 0.0,
        rel_detection_subclin = 0.0,
        tx_recovery_rate = 0.0,
        tx_relapse_rate = 0.0,
        tx_death_rate = 0.0,
        disease_mortality_clin_lowinf = 0.0,
        disease_mortality_clin_inf = 0.0,
        demography = schedule,
        ageing_enabled = true)
    fill!(p.containment_rate, 0.0)
    fill!(p.progression_rate, 0.0)
    return p
end

function _simulate_kiribati_wpp()
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR)
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
    sol = solve(prob, Rosenbrock23(autodiff = AutoFiniteDiff()); saveat = 0.25, reltol = 1e-8, abstol = 1e-10)
    return data, sol, p, u0
end

@testset "WPP data load" begin
    @test isfile(joinpath(WPP_DATA_DIR, "population.csv"))
    @test isfile(joinpath(WPP_REFERENCE_DIR, "population_2025.csv"))

    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR)
    ref = load_kiribati_agepi_reference(reference_dir = WPP_REFERENCE_DIR)

    @test data.country == WPP_COUNTRY_KIRIBATI
    @test data.years == collect(WPP_YEARS)
    @test size(data.population) == (NAGE, length(WPP_YEARS))
    @test size(data.mortality) == (NAGE, length(WPP_YEARS))
    @test size(data.fertility) == (NAGE, length(WPP_YEARS))
    @test size(data.migration) == (NAGE, length(WPP_YEARS))
    @test data.schedule.years == collect(WPP_YEARS)
    @test data.schedule.fertility_mode == :agepi_compatible
    @test data.births == [3385.0, 3370.0, 3363.0, 3363.0, 3368.0, 3376.0]

    @test isapprox(_max_abs_diff(data.population_2025, ref.population), 0.0; atol = 1e-12)
    @test isapprox(_max_abs_diff(data.mortality, ref.mortality), 0.0; atol = 1e-12)
    @test isapprox(_max_abs_diff(data.fertility, ref.fertility), 0.0; atol = 1e-12)
    @test isapprox(_max_abs_diff(data.migration, ref.migration), 0.0; atol = 1e-12)

    @test isapprox(_total_diff(data.population_2025, ref.population), 0.0; atol = 1e-12)
    @test isapprox(_total_diff(vec(data.mortality), vec(ref.mortality)), 0.0; atol = 1e-12)
    @test isapprox(_total_diff(vec(data.fertility), vec(ref.fertility)), 0.0; atol = 1e-12)
    @test isapprox(_total_diff(vec(data.migration), vec(ref.migration)), 0.0; atol = 1e-12)
end

@testset "WPP fertility modes" begin
    agepi_data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, fertility_mode = :agepi_compatible)
    wpp_data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, fertility_mode = :wpp)

    @test agepi_data.schedule.fertility_mode == :agepi_compatible
    @test wpp_data.schedule.fertility_mode == :wpp
    @test agepi_data.schedule.births == wpp_data.births
    @test wpp_data.schedule.births == wpp_data.births
    @test wpp_data.births == [3385.0, 3370.0, 3363.0, 3363.0, 3368.0, 3376.0]
    @test wpp_kiribati_demographic_schedule(data_dir = WPP_DATA_DIR, fertility_mode = :wpp).fertility_mode == :wpp
end

@testset "Fertility normalization" begin
    headers, rows = TBKiribatiJulia._read_simple_csv(joinpath(WPP_DATA_DIR, "fertility.csv"))
    idx = Dict(name => i for (i, name) in enumerate(headers))
    for year in 2025:2030
        year_rows = [r for r in rows if parse(Int, r[idx["year"]]) == year]
        pasfr_sum = sum(parse(Float64, r[idx["pasfr"]]) for r in year_rows) / 100
        tfr = parse(Float64, first(r[idx["tfr"]] for r in year_rows))
        ages = sort(parse.(Int, [r[idx["age"]] for r in year_rows]))
        fertility_rates = sum(tfr * parse(Float64, r[idx["pasfr"]]) / 100 for r in year_rows)
        @test isapprox(pasfr_sum, 1.0; atol = 1e-7)
        @test isapprox(fertility_rates, tfr; atol = 1e-7)
        @test ages == collect(10:54)
    end
end

@testset "Demography only" begin
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR)
    p = _make_zero_epi_params(data.schedule)
    u0 = initial_state(data.population_2025)
    prob = ODEProblem(tb_rhs!, u0, (2025.0, 2030.0), p)
    sol = solve(prob, Rosenbrock23(autodiff = AutoFiniteDiff()); saveat = 1.0, reltol = 1e-8, abstol = 1e-10)

    @test occursin("Success", string(sol.retcode))

    for year in 2026:2030
        idx = _time_col_index(sol.t, Float64(year))
        sim = sum(reshape(sol.u[idx], NSTATE, NAGE)[1:NEPI, :], dims = 1)[:]
        ref = data.population[:, year - 2024]
        @test all(isfinite, sim)
        @test _max_abs_diff(sim, ref) < 1.0e5
        @test _max_rel_diff(sim, ref) < 2.0
        @test abs(_total_diff(sim, ref)) < 2.0e4
    end
end

@testset "Quarterly trajectory" begin
    ref_path = joinpath(WPP_REFERENCE_DIR, "trajectory_quarterly.csv")
    @test isfile(ref_path)

    data, sol, p, u0 = _simulate_kiribati_wpp()
    ref = _reference_state_map(ref_path)
    julia = _state_map_from_solution(sol)

    @test length(ref) == length(julia)

    abs_diff = 0.0
    rel_diff = 0.0
    for (key, ref_value) in ref
        julia_value = get(julia, key, NaN)
        @test isfinite(julia_value)
        abs_diff = max(abs_diff, abs(julia_value - ref_value))
        rel_diff = max(rel_diff, abs(julia_value - ref_value) / max(abs(ref_value), eps(Float64)))
    end

    @test abs_diff < 1.0e5
    @test rel_diff < 2.0

    ref_times, ref_states = _reference_state_series(ref_path)
    julia_times, julia_states = _state_series_from_solution(sol)
    @test ref_times == julia_times

    ref_cum = _cumulative_flow_series(ref_times, ref_states, p)
    julia_cum = _cumulative_flow_series(julia_times, julia_states, p)
    @test _max_abs_diff(ref_cum, julia_cum) < 1.0e5
    @test _max_rel_diff(ref_cum, julia_cum) < 2.0

    rhs_alloc = @allocated tb_rhs!(similar(u0), u0, p, 2025.0)
    @test rhs_alloc == 0 || rhs_alloc < 32768
end
