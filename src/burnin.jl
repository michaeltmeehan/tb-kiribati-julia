struct BurnInCheckpoint
    years_elapsed::Float64
    convergence_error::Float64
    active_tb_prevalence::Float64
    annual_incidence::Float64
end

struct BurnInResult
    state::Vector{Float64}
    converged::Bool
    extinct::Bool
    years_required::Float64
    final_convergence_error::Float64
    final_active_tb_prevalence::Float64
    final_compartment_shares::NamedTuple
    history::Vector{BurnInCheckpoint}
end

struct HistoricalInitializationResult
    burnin::BurnInResult
    state_1950::Vector{Float64}
    historical_solution::Any
    target_year::Int
end

function _age_state_totals(state::AbstractVector{<:Real})
    totals = zeros(Float64, NAGE)
    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        total = 0.0
        for j in 1:NEPI
            total += state[base + j]
        end
        totals[a] = total
    end
    return totals
end

function _age_compartment_proportions(state::AbstractVector{<:Real})
    proportions = zeros(Float64, NSTATE, NAGE)
    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        total = 0.0
        for j in 1:NEPI
            total += state[base + j]
        end
        if total > 0.0
            inv_total = 1 / total
            for j in 1:NEPI
                proportions[j, a] = state[base + j] * inv_total
            end
        end
    end
    return proportions
end

function _broad_state_shares(state::AbstractVector{<:Real})
    totals = _age_state_totals(state)
    total = sum(totals)
    if total == 0.0
        return (
            naive = 0.0,
            contained = 0.0,
            cleared = 0.0,
            recovered = 0.0,
            incipient = 0.0,
            subclin_low = 0.0,
            subclin_inf = 0.0,
            clin_low = 0.0,
            clin_inf = 0.0,
            treatment = 0.0,
        )
    end

    shares = reshape(Float64.(state), NSTATE, NAGE)
    inv_total = 1 / total
    return (
        naive = sum(shares[MtbNaive, :]) * inv_total,
        contained = sum(shares[Contained, :]) * inv_total,
        cleared = sum(shares[Cleared, :]) * inv_total,
        recovered = sum(shares[Recovered, :]) * inv_total,
        incipient = sum(shares[Incipient, :]) * inv_total,
        subclin_low = sum(shares[SubClinLow, :]) * inv_total,
        subclin_inf = sum(shares[SubClinInf, :]) * inv_total,
        clin_low = sum(shares[ClinLow, :]) * inv_total,
        clin_inf = sum(shares[ClinInf, :]) * inv_total,
        treatment = sum(shares[Treatment, :]) * inv_total,
    )
end

function rebase_state_to_population(
    state::AbstractVector{<:Real},
    target_population::AbstractVector{<:Real};
    reset_cumulative::Bool = true,
)
    length(target_population) == NAGE || error("target_population must have length 96")
    u = Vector{Float64}(state)
    U = reshape(u, NSTATE, NAGE)

    @inbounds for a in 1:NAGE
        total = 0.0
        for j in 1:NEPI
            total += U[j, a]
        end
        target = Float64(target_population[a])
        if total > 0.0
            scale = target / total
            for j in 1:NEPI
                U[j, a] *= scale
            end
        else
            U[MtbNaive, a] = target
            for j in 2:NEPI
                U[j, a] = 0.0
            end
        end

        if reset_cumulative
            for j in (NEPI + 1):NSTATE
                U[j, a] = 0.0
            end
        end
    end

    return u
end

function _annual_incidence_from_solution(sol, idx::Int)
    idx <= 1 && return 0.0
    prev_state = reshape(sol.u[idx - 1], NSTATE, NAGE)
    curr_state = reshape(sol.u[idx], NSTATE, NAGE)
    prev = sum(prev_state[CumInfectionsOther:CumInfectionsContained, :])
    curr = sum(curr_state[CumInfectionsOther:CumInfectionsContained, :])
    return curr - prev
end

function run_stationary_burnin(
    p::TBParams,
    population_1950::AbstractVector{<:Real};
    max_years::Real = 150.0,
    block_years::Real = 25.0,
    convergence_tol::Real = 1e-6,
    extinction_tol::Real = 1e-12,
    seed_kwargs...,
)
    p.simulation_regime = :stationary_burnin
    p.ageing_enabled = false
    u = burnin_seed_state(population_1950; seed_kwargs...)
    prev_props = _age_compartment_proportions(u)
    history = BurnInCheckpoint[]
    elapsed = 0.0
    converged = false
    extinct = false
    final_err = Inf
    final_active = NaN

    while elapsed < max_years
        next_elapsed = min(elapsed + block_years, float(max_years))
        tspan = (elapsed, next_elapsed)
        saveat = collect(elapsed:1.0:next_elapsed)
        tstops = collect((elapsed + 1.0):1.0:next_elapsed)
        prob = ODEProblem(tb_rhs!, u, tspan, p)
        sol = solve(prob, Vern7(); saveat = saveat, reltol = 1e-8, abstol = 1e-10, tstops = tstops)
        u = Vector{Float64}(sol.u[end])

        curr_props = _age_compartment_proportions(u)
        final_err = maximum(abs.(curr_props .- prev_props))
        final_active = sum(reshape(u, NSTATE, NAGE)[Incipient:Treatment, :]) / sum(_age_state_totals(u))
        annual_incidence = _annual_incidence_from_solution(sol, length(sol.u))
        push!(history, BurnInCheckpoint(next_elapsed, final_err, final_active, annual_incidence))

        if final_active <= extinction_tol
            converged = true
            extinct = true
            elapsed = next_elapsed
            break
        end
        if final_err <= convergence_tol
            converged = true
            elapsed = next_elapsed
            break
        end

        prev_props = curr_props
        elapsed = next_elapsed
    end

    if !converged && elapsed >= max_years
        converged = false
    end

    return BurnInResult(
        u,
        converged,
        extinct,
        elapsed,
        final_err,
        final_active,
        _broad_state_shares(u),
        history,
    )
end

function generate_historical_initial_state(
    ; contact::AbstractMatrix{<:Real} = default_contact_matrix(),
    wpp_years::AbstractVector{<:Integer} = collect(WPP_YEARS),
    target_year::Integer = 2025,
    burnin_seed::NamedTuple = NamedTuple(),
    burnin_max_years::Real = 150.0,
    burnin_block_years::Real = 25.0,
    convergence_tol::Real = 1e-6,
    extinction_tol::Real = 1e-12,
    kwargs...,
)
    data = load_kiribati_wpp_data(years = wpp_years, fertility_mode = :wpp)
    burnin_params = make_parameters(contact;
        demography = nothing,
        historical = nothing,
        simulation_regime = :stationary_burnin,
        detection_rate = 0.0,
        tx_success_prop = 0.0,
        ageing_enabled = false,
        kwargs...)
    burnin_result = run_stationary_burnin(burnin_params, data.population_1950;
        max_years = burnin_max_years,
        block_years = burnin_block_years,
        convergence_tol = convergence_tol,
        extinction_tol = extinction_tol,
        burnin_seed...)

    state_1950 = rebase_state_to_population(burnin_result.state, data.population_1950; reset_cumulative = true)

    historical_years = collect(1950:Int(target_year))
    historical_schedule = wpp_kiribati_demographic_schedule(years = historical_years, fertility_mode = :wpp)
    historical_params = make_parameters(contact;
        demography = historical_schedule,
        historical = demonstration_historical_parameters(historical_years),
        simulation_regime = :historical,
        ageing_enabled = true,
        kwargs...)
    prob = ODEProblem(tb_rhs!, state_1950, (1950.0, Float64(target_year)), historical_params)
    sol = solve(prob, Vern7(); reltol = 1e-8, abstol = 1e-10, saveat = 1950.0:1.0:Float64(target_year), tstops = demographic_tstops(historical_schedule))

    return HistoricalInitializationResult(burnin_result, state_1950, sol, Int(target_year))
end
