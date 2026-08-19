using Test
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7

const WPP_DATA_DIR = joinpath(@__DIR__, "..", "data", "wpp2024")

function zero_demographic_schedule(years::AbstractVector{<:Integer} = 2025:2030)
    nyears = length(years)
    DemographicSchedule(Int[years...], zeros(Float64, NAGE, nyears), zeros(Float64, NAGE, nyears), zeros(Float64, NAGE, nyears))
end

function demo_age_base(age::Integer)
    age * NSTATE
end

function epi_total(du)
    sum(reshape(du, NSTATE, NAGE)[1:NEPI, :])
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

function accounting_rhs!(dX, X, p, t)
    u = @view X[1:(NSTATE * NAGE)]
    du = @view dX[1:(NSTATE * NAGE)]
    tb_rhs!(du, u, p, t)

    year_idx = TBKiribatiJulia.demographic_year_index(p.demography, t)
    age_pop = zeros(Float64, NAGE)
    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        total = 0.0
        for j in 1:NEPI
            total += u[base + j]
        end
        age_pop[a] = total
    end

    deaths = 0.0
    migration = 0.0
    @inbounds for a in 1:NAGE
        deaths += p.demography.mortality[a, year_idx] * age_pop[a]
        migration += p.demography.migration[a, year_idx]
    end

    dX[NSTATE * NAGE + 1] = TBKiribatiJulia.demographic_births(p.demography, year_idx, age_pop)
    dX[NSTATE * NAGE + 2] = deaths
    dX[NSTATE * NAGE + 3] = migration
    return nothing
end

@testset "Demographic regression" begin
    schedule = zero_demographic_schedule()
    p = zero_transition_params(; demography = schedule, ageing_enabled = false)
    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[Recovered, age_index(12)] = 25.0
    du_epi = similar(u)
    du_full = similar(u)
    fill!(du_epi, 0.0)
    tb_rhs_epi!(du_epi, u, p, 2025.0)
    tb_rhs!(du_full, u, p, 2025.0)
    @test du_epi == du_full
end

@testset "Ageing" begin
    schedule = zero_demographic_schedule()
    p = zero_transition_params(; demography = schedule, ageing_enabled = true)
    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(0)] = 100.0
    U[MtbNaive, age_index(1)] = 50.0
    U[MtbNaive, age_index(94)] = 25.0
    U[MtbNaive, age_index(95)] = 10.0

    du = similar(u)
    tb_rhs!(du, u, p, 2025.2)
    DU = reshape(du, NSTATE, NAGE)

    @test isapprox(epi_total(du), 0.0; atol = 1e-12, rtol = 1e-12)
    @test DU[MtbNaive, age_index(0)] == -100.0
    @test DU[MtbNaive, age_index(1)] == 50.0
    @test DU[MtbNaive, age_index(94)] == -25.0
    @test DU[MtbNaive, age_index(95)] == 25.0
end

@testset "Open-ended age group" begin
    schedule = zero_demographic_schedule()
    p = zero_transition_params(; demography = schedule, ageing_enabled = true)
    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(95)] = 123.0

    du = similar(u)
    tb_rhs!(du, u, p, 2025.2)
    DU = reshape(du, NSTATE, NAGE)
    @test DU[MtbNaive, age_index(95)] == 0.0
end

@testset "Background mortality" begin
    years = [2025]
    mortality = zeros(Float64, NAGE, 1)
    fertility = zeros(Float64, NAGE, 1)
    migration = zeros(Float64, NAGE, 1)
    mortality[age_index(10), 1] = 0.02
    mortality[age_index(40), 1] = 0.03
    schedule = DemographicSchedule(years, mortality, fertility, migration)
    p = zero_transition_params(; demography = schedule, ageing_enabled = false)

    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(10)] = 100.0
    U[Contained, age_index(10)] = 50.0
    U[Recovered, age_index(40)] = 20.0
    U[ClinInf, age_index(40)] = 5.0

    du = similar(u)
    tb_rhs!(du, u, p, 2025.1)
    DU = reshape(du, NSTATE, NAGE)

    @test isapprox(DU[MtbNaive, age_index(10)], -2.0; atol = 1e-12)
    @test isapprox(DU[Contained, age_index(10)], -1.0; atol = 1e-12)
    @test isapprox(DU[Recovered, age_index(40)], -0.6; atol = 1e-12)
    @test isapprox(DU[ClinInf, age_index(40)], -0.15; atol = 1e-12)
end

@testset "Births" begin
    years = [2025]
    mortality = zeros(Float64, NAGE, 1)
    fertility = zeros(Float64, NAGE, 1)
    migration = zeros(Float64, NAGE, 1)
    fertility[age_index(15), 1] = 0.1
    fertility[age_index(30), 1] = 0.2
    schedule = DemographicSchedule(years, mortality, fertility, migration)
    p = zero_transition_params(; demography = schedule, ageing_enabled = false)

    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(15)] = 100.0
    U[Contained, age_index(15)] = 50.0
    U[Recovered, age_index(30)] = 20.0

    du = similar(u)
    tb_rhs!(du, u, p, 2025.0)
    DU = reshape(du, NSTATE, NAGE)

    @test isapprox(DU[MtbNaive, age_index(0)], 19.0; atol = 1e-12)
    @test all(iszero, DU[2:NEPI, age_index(0)])
end

@testset "Fertility mode selection" begin
    years = [2025]
    mortality = zeros(Float64, NAGE, 1)
    fertility = zeros(Float64, NAGE, 1)
    migration = zeros(Float64, NAGE, 1)
    fertility[age_index(15), 1] = 0.1
    fertility[age_index(30), 1] = 0.2

    agepi_schedule = DemographicSchedule(years, mortality, fertility, migration; births = [12345.0], fertility_mode = :agepi_compatible)
    wpp_schedule = DemographicSchedule(years, mortality, fertility, migration; births = [321.0], fertility_mode = :wpp)
    p_agepi = zero_transition_params(; demography = agepi_schedule, ageing_enabled = false)
    p_wpp = zero_transition_params(; demography = wpp_schedule, ageing_enabled = false)

    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(15)] = 100.0
    U[Contained, age_index(15)] = 50.0
    U[Recovered, age_index(30)] = 20.0

    du = similar(u)
    tb_rhs!(du, u, p_agepi, 2025.0)
    DU = reshape(du, NSTATE, NAGE)
    @test isapprox(DU[MtbNaive, age_index(0)], 19.0; atol = 1e-12)

    fill!(du, 0.0)
    tb_rhs!(du, u, p_wpp, 2025.0)
    DU = reshape(du, NSTATE, NAGE)
    @test isapprox(DU[MtbNaive, age_index(0)], 321.0; atol = 1e-12)
    @test wpp_schedule.fertility_mode == :wpp
    @test agepi_schedule.fertility_mode == :agepi_compatible
end

@testset "Migration" begin
    years = [2025]
    mortality = zeros(Float64, NAGE, 1)
    fertility = zeros(Float64, NAGE, 1)
    migration = zeros(Float64, NAGE, 1)
    migration[age_index(20), 1] = 20.0
    schedule = DemographicSchedule(years, mortality, fertility, migration)
    p = zero_transition_params(; demography = schedule, ageing_enabled = false)

    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(20)] = 60.0
    U[Contained, age_index(20)] = 30.0
    U[Recovered, age_index(20)] = 10.0

    du = similar(u)
    tb_rhs!(du, u, p, 2025.0)
    DU = reshape(du, NSTATE, NAGE)

    @test isapprox(DU[MtbNaive, age_index(20)], 12.0; atol = 1e-12)
    @test isapprox(DU[Contained, age_index(20)], 6.0; atol = 1e-12)
    @test isapprox(DU[Recovered, age_index(20)], 2.0; atol = 1e-12)

    migration[age_index(20), 1] = -20.0
    schedule = DemographicSchedule(years, mortality, fertility, migration)
    p = zero_transition_params(; demography = schedule, ageing_enabled = false)
    fill!(du, 0.0)
    tb_rhs!(du, u, p, 2025.0)
    DU = reshape(du, NSTATE, NAGE)

    @test isapprox(DU[MtbNaive, age_index(20)], -12.0; atol = 1e-12)
    @test isapprox(DU[Contained, age_index(20)], -6.0; atol = 1e-12)
    @test isapprox(DU[Recovered, age_index(20)], -2.0; atol = 1e-12)
end

@testset "Migration into empty age group" begin
    years = [2025]
    mortality = zeros(Float64, NAGE, 1)
    fertility = zeros(Float64, NAGE, 1)
    migration = zeros(Float64, NAGE, 1)
    migration[age_index(30), 1] = 8.0
    schedule = DemographicSchedule(years, mortality, fertility, migration)
    p = zero_transition_params(; demography = schedule, ageing_enabled = false)

    u = blank_state()
    du = similar(u)
    tb_rhs!(du, u, p, 2025.0)
    DU = reshape(du, NSTATE, NAGE)
    @test isapprox(DU[MtbNaive, age_index(30)], 8.0; atol = 1e-12)
    @test all(iszero, DU[2:NEPI, age_index(30)])

    migration[age_index(30), 1] = -8.0
    schedule = DemographicSchedule(years, mortality, fertility, migration)
    p = zero_transition_params(; demography = schedule, ageing_enabled = false)
    fill!(du, 0.0)
    tb_rhs!(du, u, p, 2025.0)
    DU = reshape(du, NSTATE, NAGE)
    @test all(iszero, DU[1:NEPI, age_index(30)])
end

@testset "Annual step lookup" begin
    years = [2025, 2026]
    mortality = zeros(Float64, NAGE, 2)
    fertility = zeros(Float64, NAGE, 2)
    migration = zeros(Float64, NAGE, 2)
    mortality[age_index(20), 1] = 0.10
    mortality[age_index(20), 2] = 0.20
    schedule = DemographicSchedule(years, mortality, fertility, migration)
    p = zero_transition_params(; demography = schedule, ageing_enabled = false)

    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(20)] = 100.0
    du = similar(u)

    for t in (2025.0, 2025.25, 2025.999999)
        fill!(du, 0.0)
        tb_rhs!(du, u, p, t)
        @test isapprox(du[demo_age_base(20) + MtbNaive], -10.0; atol = 1e-12)
    end

    for t in (2026.0, 2026.25)
        fill!(du, 0.0)
        tb_rhs!(du, u, p, t)
        @test isapprox(du[demo_age_base(20) + MtbNaive], -20.0; atol = 1e-12)
    end
end

@testset "Demographic tstops" begin
    schedule = zero_demographic_schedule()
    @test TBKiribatiJulia.demographic_tstops(schedule) == Float64.(2026:2030)
end

@testset "Annual demographic accounting" begin
    years = 2025:2030
    schedule = zero_demographic_schedule(years)
    for yi in 1:length(years)
        year = years[yi]
        schedule.fertility[16, yi] = 0.10
        schedule.fertility[31, yi] = 0.20
        schedule.mortality[11, yi] = 0.01
        schedule.mortality[41, yi] = 0.02
        schedule.migration[21, yi] = -3.0
        schedule.migration[31, yi] = 2.0
        schedule.fertility[16, yi] += 0.0 * year
    end
    p = zero_transition_params(; demography = schedule, ageing_enabled = true)
    u0 = initial_state(fill(1000.0, NAGE))
    x0 = vcat(u0, zeros(Float64, 3))
    prob = ODEProblem(accounting_rhs!, x0, (2025.0, 2030.0), p)
    sol = solve(prob, Vern7(); saveat = 2025.0:1.0:2030.0, reltol = 1e-10, abstol = 1e-12, tstops = TBKiribatiJulia.demographic_tstops(schedule))

    for year in 2025:2029
        start_idx = findfirst(==(Float64(year)), sol.t)
        end_idx = findfirst(==(Float64(year + 1)), sol.t)
        start_idx === nothing && error("missing start year $year")
        end_idx === nothing && error("missing end year $(year + 1)")
        u_start = reshape(sol.u[start_idx][1:(NSTATE * NAGE)], NSTATE, NAGE)
        u_end = reshape(sol.u[end_idx][1:(NSTATE * NAGE)], NSTATE, NAGE)
        n_start = sum(u_start[1:NEPI, :])
        n_end = sum(u_end[1:NEPI, :])
        births = sol.u[end_idx][NSTATE * NAGE + 1] - sol.u[start_idx][NSTATE * NAGE + 1]
        deaths = sol.u[end_idx][NSTATE * NAGE + 2] - sol.u[start_idx][NSTATE * NAGE + 2]
        migration = sol.u[end_idx][NSTATE * NAGE + 3] - sol.u[start_idx][NSTATE * NAGE + 3]
        @test isapprox(n_end - n_start, births - deaths + migration; atol = 1e-8, rtol = 1e-8)

        age0_start = sum(u_start[1:NEPI, 1])
        age0_end = sum(u_end[1:NEPI, 1])
        age0_births = births
        year_col = min(year - first(years) + 1, size(schedule.migration, 2))
        age0_migration = schedule.migration[1, year_col]
        age0_deaths = schedule.mortality[1, year_col] * sum(u_start[:, 1])
        age0_ageing_out = age0_births + age0_migration - age0_deaths - (age0_end - age0_start)
        @test isapprox(age0_end - age0_start, age0_births + age0_migration - age0_deaths - age0_ageing_out; atol = 1e-8, rtol = 1e-8)
    end
end

@testset "WPP demographic accounting" begin
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, fertility_mode = :wpp)
    schedule = data.schedule
    p = _make_zero_epi_params(schedule)
    u0 = initial_state(data.population_2025)
    x0 = vcat(u0, zeros(Float64, 3))
    prob = ODEProblem(accounting_rhs!, x0, (2025.0, 2030.0), p)
    sol = solve(prob, Vern7(); saveat = 2025.0:1.0:2030.0, reltol = 1e-10, abstol = 1e-12, tstops = TBKiribatiJulia.demographic_tstops(schedule))

    for year in 2025:2029
        start_idx = findfirst(==(Float64(year)), sol.t)
        end_idx = findfirst(==(Float64(year + 1)), sol.t)
        start_idx === nothing && error("missing start year $year")
        end_idx === nothing && error("missing end year $(year + 1)")
        u_start = reshape(sol.u[start_idx][1:(NSTATE * NAGE)], NSTATE, NAGE)
        u_end = reshape(sol.u[end_idx][1:(NSTATE * NAGE)], NSTATE, NAGE)
        n_start = sum(u_start[1:NEPI, :])
        n_end = sum(u_end[1:NEPI, :])
        births = sol.u[end_idx][NSTATE * NAGE + 1] - sol.u[start_idx][NSTATE * NAGE + 1]
        deaths = sol.u[end_idx][NSTATE * NAGE + 2] - sol.u[start_idx][NSTATE * NAGE + 2]
        migration = sol.u[end_idx][NSTATE * NAGE + 3] - sol.u[start_idx][NSTATE * NAGE + 3]
        expected_births = data.births[year - 2024]

        @test isapprox(births, expected_births; atol = 1e-8, rtol = 1e-8)
        @test isapprox(n_end - n_start, births - deaths + migration; atol = 1e-8, rtol = 1e-8)

        age0_start = sum(u_start[1:NEPI, 1])
        age0_end = sum(u_end[1:NEPI, 1])
        age0_migration = schedule.migration[1, year - 2024]
        age0_deaths = schedule.mortality[1, year - 2024] * sum(u_start[:, 1])
        age0_ageing_out = births + age0_migration - age0_deaths - (age0_end - age0_start)
        @test isapprox(age0_end - age0_start, births + age0_migration - age0_deaths - age0_ageing_out; atol = 1e-8, rtol = 1e-8)
    end
end

@testset "Combined demographic balance" begin
    years = [2025]
    mortality = zeros(Float64, NAGE, 1)
    fertility = zeros(Float64, NAGE, 1)
    migration = zeros(Float64, NAGE, 1)
    mortality[age_index(10), 1] = 0.01
    mortality[age_index(20), 1] = 0.02
    mortality[age_index(30), 1] = 0.03
    fertility[age_index(20), 1] = 0.10
    fertility[age_index(30), 1] = 0.20
    migration[age_index(10), 1] = 5.0
    migration[age_index(20), 1] = -3.0
    migration[age_index(30), 1] = 2.0
    schedule = DemographicSchedule(years, mortality, fertility, migration)
    p = zero_transition_params(; demography = schedule, ageing_enabled = true)

    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[MtbNaive, age_index(10)] = 100.0
    U[Contained, age_index(10)] = 50.0
    U[Recovered, age_index(20)] = 80.0
    U[ClinInf, age_index(30)] = 40.0

    du = similar(u)
    tb_rhs!(du, u, p, 2025.5)
    DU = reshape(du, NSTATE, NAGE)

    age_pop = zeros(Float64, NAGE)
    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        age_pop[a] = sum(@view u[base + 1:base + NEPI])
    end
    births = sum(fertility[:, 1] .* age_pop)
    migration_total = sum(migration[:, 1])
    mortality_total = sum(mortality[:, 1] .* age_pop)

    @test isapprox(sum(DU[1:NEPI, :]), births + migration_total - mortality_total; atol = 1e-12, rtol = 1e-12)
end

@testset "Epidemiology plus demography" begin
    sol = simulate_demographic_demo(tspan = (2025.0, 2026.0), saveat = 0.25)
    @test occursin("Success", string(sol.retcode))

    initial = reshape(sol.u[1], NSTATE, NAGE)
    final = reshape(sol.u[end], NSTATE, NAGE)
    @test !isapprox(sum(initial[1:NEPI, :]), sum(final[1:NEPI, :]); atol = 1e-8)
    @test !isapprox(sum(initial[:, age_index(95)]), sum(final[:, age_index(95)]); atol = 1e-8)

    for state in sol.u
        @test all(isfinite, state)
        epi = reshape(state, NSTATE, NAGE)
        @test minimum(epi[1:NEPI, :]) ≥ -1e-8
    end

    for idx in CumInfectionsOther:CumRelapseTB
        series = [sum(reshape(state, NSTATE, NAGE)[idx, :]) for state in sol.u]
        @test all(diff(series) .>= -1e-10)
    end

    @test sum(final[:, age_index(95)]) != sum(initial[:, age_index(95)])
end
