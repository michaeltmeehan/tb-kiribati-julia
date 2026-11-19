
using LinearAlgebra: norm
using SciMLBase: successful_retcode

"""
    equilibrium_parameter_check(model, params)

Validate that model parameters define time-independent
continuous-time dynamics during equilibration.

Each model must implement this explicitly.
"""
function equilibrium_parameter_check end

function equilibrium_parameter_check(
    ::DoddTBModel,
    params::DoddTBParameters,
)
    iszero(params.detection.trend) ||
        throw(ArgumentError(
            "Equilibrium requires detection.trend == 0"
        ))

    return nothing
end

function equilibrium_parameter_check(
    ::DoddExtendedTBModel,
    params::DoddExtendedTBParameters,
)
    iszero(params.detection.trend) ||
        throw(ArgumentError(
            "Equilibrium requires detection.trend == 0"
        ))

    return nothing
end

function equilibrium_parameter_check(
    model::AbstractTBModel,
    params,
)
    throw(ArgumentError(
        "No equilibrium stationarity check defined for " *
            string(typeof(model))
    ))
end


"""
    equilibrate(model, params, demography; u0,
                start_year, block_years=10,
                max_years=500,
                state_tol=1e-6,
                incidence_tol=1e-6)

Approximate a fixed point of the annual demographic +
epidemiological update.

The convergence diagnostics compare:
- successive post-demographic living states;
- annual disease-onset counts in successive years.

Returns a state with cumulative event counters reset.

Convergence is not guaranteed, and a converged result
is not necessarily the unique equilibrium.
"""
function equilibrate(
    model::AbstractTBModel,
    params,
    demography::EquilibriumDemography;
    u0::AbstractVector,
    start_year::Integer=demography.reference_year,
    block_years::Integer=10,
    max_years::Integer=500,
    state_tol::Real=1e-6,
    incidence_tol::Real=1e-6,
    solver=Tsit5(),
    abstol::Real=1e-8,
    reltol::Real=1e-6,
)
    equilibrium_parameter_check(model, params)

    block_years > 0 ||
        throw(ArgumentError("block_years must be positive"))
    max_years > 0 ||
        throw(ArgumentError("max_years must be positive"))
    state_tol > 0 ||
        throw(ArgumentError("state_tol must be positive"))
    incidence_tol > 0 ||
        throw(ArgumentError("incidence_tol must be positive"))

    layout = state_layout(model)
    length(u0) == state_dimension(layout) ||
        throw(DimensionMismatch("Invalid initial state length"))

    all(isfinite, u0) ||
        throw(ArgumentError("Nonfinite initial state"))

    stride = nstates(layout)

    epi_idx = [
        state_index(layout, a, c)
        for a in 1:layout.nages
        for c in 1:layout.nepi
    ]

    counter_idx = [
        state_index(layout, a, c)
        for a in 1:layout.nages
        for c in (layout.nepi+1):stride
    ]

    u = copy(u0)
    elapsed = 0

    previous_incidence = nothing

    state_change = Inf
    incidence_change = Inf
    annual_incidence = NaN
    converged = false

    while elapsed < max_years
        nblock = min(
            block_years,
            max_years - elapsed,
        )

        first_year = Int(start_year + elapsed)
        final_year = first_year + nblock

        setup = prepare_simulation(
            model,
            demography;
            start_year=first_year,
            end_year=final_year,
            solver,
            abstol,
            reltol,
        )

        sol = simulate(setup, params; u0=u)

        successful_retcode(sol) ||
            error("Equilibrium simulation failed")

        years = collect(first_year:(final_year-1))
        windows = annual_windows(sol; years)

        # One model-specific output requested, not the
        # entire model output collection.
        incidence = model_output(
            model,
            sol,
            :disease_onsets;
            years,
        )

        for j in eachindex(years)
            before = sol.u[windows.start_indices[j]]

            # The last saved state at year+1 is the
            # post-demographic state.
            boundary = years[j] + 1
            after_idx = findlast(
                t -> abs(t - boundary) <= 1e-7,
                sol.t,
            )

            isnothing(after_idx) &&
                error("Missing post-demographic state")

            after = sol.u[after_idx]

            state_change = norm(
                after[epi_idx] - before[epi_idx], 1
            ) / max(
                norm(before[epi_idx], 1),
                eps(Float64),
            )

            annual_incidence = sum(@view incidence[:, j])

            if !isnothing(previous_incidence)
                incidence_change = abs(
                    annual_incidence - previous_incidence
                ) / max(
                    abs(previous_incidence),
                    eps(Float64),
                )
            end

            elapsed += 1

            if state_change < state_tol &&
                incidence_change < incidence_tol
                u = copy(after)
                converged = true
                break
            end

            previous_incidence = annual_incidence
        end

        converged && break

        u = copy(sol.u[end])
    end

    # Counters accumulate events during equilibration
    # but should not carry into subsequent projections.
    u[counter_idx] .= zero(eltype(u))

    return (
        u=u,
        converged=converged,
        years=elapsed,
        incidence=annual_incidence,
        state_change=state_change,
        incidence_change=incidence_change,
    )
end
