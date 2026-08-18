using Test

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
