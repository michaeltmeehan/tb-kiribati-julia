using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7
using TBKiribatiJulia

const WPP_DATA_DIR = joinpath(@__DIR__, "..", "data", "wpp2024")
const NEXTRA = 6

function build_problem()
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR)
    u0 = initial_state(data.population_2025)
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
        demography = data.schedule,
        ageing_enabled = true)
    x0 = vcat(u0, zeros(Float64, NEXTRA))
    prob = ODEProblem(augmented_rhs!, x0, (2025.0, 2030.0), p)
    return data, p, prob
end

function age_population(state)
    epi = reshape(state, NSTATE, NAGE)
    age_pop = zeros(Float64, NAGE)
    @inbounds for a in 1:NAGE
        age_pop[a] = sum(@view epi[1:NEPI, a])
    end
    return age_pop
end

function demographic_rates(state, p, t)
    epi = reshape(state, NSTATE, NAGE)
    age_pop = age_population(state)
    year_idx = TBKiribatiJulia.demographic_year_index(p.demography, t)
    births = 0.0
    deaths = 0.0
    migration = 0.0
    age0_migration = 0.0
    age0_deaths = 0.0
    age0_ageing_out = 0.0
    @inbounds for a in 1:NAGE
        fert = p.demography.fertility[a, year_idx]
        mu = p.demography.mortality[a, year_idx]
        mig = p.demography.migration[a, year_idx]
        births += fert * age_pop[a]
        deaths += mu * age_pop[a]
        migration += mig
        if a == 1
            age0_migration = mig
            age0_deaths = mu * age_pop[a]
            if p.ageing_enabled
                age0_ageing_out = sum(@view epi[1:NEPI, a])
            end
        end
    end
    return (; births, deaths, migration, age0_migration, age0_deaths, age0_ageing_out, age0_pop = age_pop[1])
end

function augmented_rhs!(dX, X, p, t)
    u = @view X[1:(NSTATE * NAGE)]
    du = @view dX[1:(NSTATE * NAGE)]
    tb_rhs!(du, u, p, t)
    r = demographic_rates(u, p, t)
    dX[NSTATE * NAGE + 1] = r.births
    dX[NSTATE * NAGE + 2] = r.deaths
    dX[NSTATE * NAGE + 3] = r.migration
    dX[NSTATE * NAGE + 4] = r.births
    dX[NSTATE * NAGE + 5] = r.age0_deaths
    dX[NSTATE * NAGE + 6] = r.age0_migration
    return nothing
end

function annual_accounting(sol)
    rows = NamedTuple[]
    for year in 2025:2029
        start_idx = findfirst(==(Float64(year)), sol.t)
        end_idx = findfirst(==(Float64(year + 1)), sol.t)
        start_idx === nothing && error("missing start year $year")
        end_idx === nothing && error("missing end year $(year + 1)")
        u_start = view(sol.u[start_idx], 1:(NSTATE * NAGE))
        u_end = view(sol.u[end_idx], 1:(NSTATE * NAGE))
        N_start = sum(age_population(u_start))
        N_end = sum(age_population(u_end))
        B = sol.u[end_idx][NSTATE * NAGE + 1] - sol.u[start_idx][NSTATE * NAGE + 1]
        D = sol.u[end_idx][NSTATE * NAGE + 2] - sol.u[start_idx][NSTATE * NAGE + 2]
        M = sol.u[end_idx][NSTATE * NAGE + 3] - sol.u[start_idx][NSTATE * NAGE + 3]
        age0_births = sol.u[end_idx][NSTATE * NAGE + 4] - sol.u[start_idx][NSTATE * NAGE + 4]
        age0_deaths = sol.u[end_idx][NSTATE * NAGE + 5] - sol.u[start_idx][NSTATE * NAGE + 5]
        age0_migration = sol.u[end_idx][NSTATE * NAGE + 6] - sol.u[start_idx][NSTATE * NAGE + 6]
        age0_start = sum(view(reshape(u_start, NSTATE, NAGE), 1:NEPI, 1))
        age0_end = sum(view(reshape(u_end, NSTATE, NAGE), 1:NEPI, 1))
        age0_ageing_out = age0_births + age0_migration - age0_deaths - (age0_end - age0_start)
        delta = N_end - N_start
        residual = delta - (B - D + M)
        push!(rows, (; year, N_start, N_end, delta, B, D, M, balance = B - D + M, residual,
            age0_start, age0_end, age0_births, age0_migration, age0_deaths, age0_ageing_out))
    end
    return rows
end

function main()
    data, p, prob = build_problem()
    sol = solve(prob, Vern7(); saveat = 2025.0:1.0:2030.0, reltol = 1e-8, abstol = 1e-10, tstops = 2026.0:1.0:2030.0)
    rows = annual_accounting(sol)
    for row in rows
        println(row)
    end
end

main()
