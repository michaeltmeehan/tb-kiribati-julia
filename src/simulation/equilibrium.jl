using LinearAlgebra: norm

"""
    equilibrate(model, params, demography;
                u0, start_year=2023, block_years=10,
                max_years=500, state_tol=1e-6,
                incidence_tol=1e-6)

Numerically approximate the stationary annual epidemiological
state under fixed demography and time-independent parameters.

Returns a named tuple containing the equilibrium initial state,
annual incidence, elapsed years and convergence diagnostics.

Cumulative event counters are reset to zero in the returned state.
"""
function equilibrate(
    model::DoddTBModel,
    params::DoddTBParameters,
    demography::EquilibriumDemography;
    u0::AbstractVector,
    start_year::Int = 2023,
    block_years::Int = 10,
    max_years::Int = 500,
    state_tol::Real = 1e-6,
    incidence_tol::Real = 1e-6,
)

iszero(params.detection.trend) ||
    throw(ArgumentError(
        "Equilibration requires a constant detection probability " *
        "(detection.trend == 0)."
    ))

block_years > 0 ||
    throw(ArgumentError("block_years must be positive"))

max_years > 0 ||
    throw(ArgumentError("max_years must be positive"))

state_tol > 0 ||
    throw(ArgumentError("state_tol must be positive"))

incidence_tol > 0 ||
    throw(ArgumentError("incidence_tol must be positive"))
    
    block_years > 0 || throw(ArgumentError(
        "block_years must be positive"
    ))
    max_years > 0 || throw(ArgumentError(
        "max_years must be positive"
    ))

    layout = state_layout(model)
    nages = layout.nages
    nepi = layout.nepi
    ns = nstates(layout)

    epi_idx = [
        state_index(layout, a, c)
        for a in 1:nages
        for c in 1:nepi
    ]

    # Dodd-specific cumulative incidence counters.
    incidence_idx = [
        state_index(layout, a, c)
        for a in 1:nages
        for c in (
            CumFastProgression,
            CumSlowProgression,
            CumRelapse,
        )
    ]

    u = copy(u0)
    elapsed = 0

    previous_incidence = NaN
    state_change = Inf
    incidence_change = Inf
    annual_incidence = NaN
    converged = false

    while elapsed < max_years

        years = min(block_years, max_years - elapsed)

        t0 = Float64(start_year + elapsed)
        t1 = t0 + years

        sol = simulate(
            model,
            params,
            demography;
            u0 = u,
            tspan = (t0, t1),
        )

        SciMLBase.successful_retcode(sol) ||
            error("Equilibrium simulation failed")

        for j in 2:length(sol.u)

            previous = sol.u[j - 1]
            current = sol.u[j]

            state_change = norm(
                current[epi_idx] - previous[epi_idx], 1
            ) / max(norm(previous[epi_idx], 1), eps())

            current_incidence = sum(current[incidence_idx])
            previous_cumulative = sum(previous[incidence_idx])

            annual_incidence =
                current_incidence - previous_cumulative

            if isfinite(previous_incidence)
                incidence_change = abs(
                    annual_incidence - previous_incidence
                ) / max(abs(previous_incidence), eps())
            end

            if state_change < state_tol &&
               incidence_change < incidence_tol

                u = copy(current)
                elapsed += j - 1
                converged = true
                break
            end

            previous_incidence = annual_incidence
        end

        converged && break

        u = copy(sol.u[end])
        elapsed += years
    end

    # Reset cumulative counters for downstream projections.
    for a in 1:nages
        for c in (nepi + 1):ns
            u[state_index(layout, a, c)] = 0.0
        end
    end

    return (
        u = u,
        converged = converged,
        years = elapsed,
        incidence = annual_incidence,
        state_change = state_change,
        incidence_change = incidence_change,
    )
end