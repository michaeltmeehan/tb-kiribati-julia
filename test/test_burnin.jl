using Test

const WPP_DATA_DIR = joinpath(@__DIR__, "..", "data", "wpp2024")

function _burnin_params(; beta = 1e-3)
    return make_parameters(default_contact_matrix();
        beta = beta,
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
        detection_rate = 0.0,
        rel_detection_subclin = 0.0,
        tx_recovery_rate = 0.0,
        tx_relapse_rate = 0.0,
        tx_death_rate = 0.0,
        disease_mortality_clin_lowinf = 0.025,
        disease_mortality_clin_inf = 0.4,
        simulation_regime = :stationary_burnin,
        ageing_enabled = false)
end

@testset "Stationary burn-in balances age totals" begin
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, years = 1950:2030, fertility_mode = :wpp)
    p = _burnin_params()
    u = burnin_seed_state(data.population_1950)
    du = similar(u)
    tb_rhs!(du, u, p, 0.0)
    DU = reshape(du, NSTATE, NAGE)

    for age in 1:NAGE
        @test isapprox(sum(DU[1:NEPI, age]), 0.0; atol = 1e-12, rtol = 1e-12)
    end
end

@testset "Rebase" begin
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, years = 1950:2030, fertility_mode = :wpp)
    u = burnin_seed_state(fill(1000.0, NAGE))
    U = reshape(u, NSTATE, NAGE)
    for a in 1:NAGE
        U[CumInfectionsOther, a] = 1.0
        U[CumTreatmentCompletion, a] = 2.0
    end

    rebased = rebase_state_to_population(u, data.population_1950)
    R = reshape(rebased, NSTATE, NAGE)
    for age in 1:NAGE
        @test isapprox(sum(R[1:NEPI, age]), data.population_1950[age]; atol = 1e-10, rtol = 1e-10)
        @test all(iszero, R[NEPI+1:NSTATE, age])
    end
end

@testset "Burn-in convergence and extinction" begin
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, years = 1950:2030, fertility_mode = :wpp)
    p = _burnin_params(; beta = 0.0)
    result = run_stationary_burnin(p, data.population_1950; max_years = 150.0, block_years = 25.0, convergence_tol = 1e-5, extinction_tol = 1e-5)
    @test result.converged
    @test result.extinct
    @test result.years_required <= 150.0
    @test result.final_active_tb_prevalence <= 1e-5
end

@testset "Maximum burn-in" begin
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, years = 1950:2030, fertility_mode = :wpp)
    p = _burnin_params()
    result = run_stationary_burnin(p, data.population_1950; max_years = 25.0, block_years = 25.0, convergence_tol = -1.0, extinction_tol = 0.0)
    @test !result.converged
    @test result.years_required == 25.0
end

@testset "Seed sensitivity" begin
    data = load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, years = 1950:2030, fertility_mode = :wpp)
    p1 = _burnin_params(; beta = 0.0)
    p2 = _burnin_params(; beta = 0.0)
    r1 = run_stationary_burnin(p1, data.population_1950; max_years = 150.0, block_years = 25.0, convergence_tol = 1e-5, extinction_tol = 1e-5)
    p3 = _burnin_params(; beta = 0.01)
    r2 = run_stationary_burnin(p3, data.population_1950; max_years = 150.0, block_years = 25.0, convergence_tol = 1e-5, extinction_tol = 1e-5,
        naive_share = 0.985, contained_share = 0.005, incipient_share = 0.005, clin_inf_share = 0.005)
    shares1 = collect(values(r1.final_compartment_shares))
    shares2 = collect(values(r2.final_compartment_shares))
    @test r1.converged
    @test isapprox(r1.final_active_tb_prevalence, r2.final_active_tb_prevalence; atol = 5e-6, rtol = 0.2)
    @test maximum(abs.(shares1 .- shares2)) < 5e-3
end

@testset "Historical integration" begin
    result = generate_historical_initial_state(
        contact = default_contact_matrix(),
        wpp_years = 1950:2025,
        target_year = 2025,
        burnin_max_years = 25.0,
        burnin_block_years = 25.0,
        convergence_tol = 1e-6,
        extinction_tol = 1e-8,
        beta = 1e-3)

    @test result.target_year == 2025
    @test occursin("Success", string(result.historical_solution.retcode))
    @test result.burnin.years_required <= 25.0
    @test isapprox(sum(reshape(result.state_1950, NSTATE, NAGE)[1:NEPI, :], dims = 1)[:], load_kiribati_wpp_data(data_dir = WPP_DATA_DIR, years = 1950:2025, fertility_mode = :wpp).population_1950; atol = 1e-8, rtol = 1e-8)
    @test all(iszero, reshape(result.state_1950, NSTATE, NAGE)[NEPI+1:NSTATE, :])
    @test all(isfinite, result.historical_solution.u[end])
end
