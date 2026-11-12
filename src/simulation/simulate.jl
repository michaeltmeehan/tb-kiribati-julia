# src/simulation/simulate.jl

using OrdinaryDiffEq
using DiffEqCallbacks: PresetTimeCallback
using SciMLBase: ODEProblem, solve, u_modified!

"""
    simulate(model, params, demography; u0, tspan, saveat, ...)

Simulate an age-structured TB model with annual demographic updates.

Time is expressed in calendar years.

At each integer-year boundary after the initial year,
the demographic operator advances the population from
the preceding year to the current year.

Saved states at demographic update times are post-update.
"""
function simulate(
    model::AbstractTBModel,
    params,
    demography::AbstractDemography;
    u0::AbstractVector,
    tspan::Tuple{<:Real,<:Real},
    saveat = nothing,
    solver = Tsit5(),
    abstol::Real = 1e-8,
    reltol::Real = 1e-6,
)
    layout = state_layout(model)

    length(u0) == layout.nages * nstates(layout) ||
        throw(DimensionMismatch(
            "Initial state length does not match model layout"
        ))

    t0, t1 = Float64.(tspan)

    t0 < t1 ||
        throw(ArgumentError("tspan must have positive duration"))

    isinteger(t0) ||
        throw(ArgumentError(
            "Simulation must begin at an integer calendar year"
        ))

    all(isfinite, u0) ||
        throw(ArgumentError("Initial state contains nonfinite values"))

    # --------------------------------------------------------
    # Simulation context
    # --------------------------------------------------------

    workspace = TBWorkspace(layout.nages)

    context = SimulationContext(
        model,
        params,
        demography,
        workspace,
    )

    # --------------------------------------------------------
    # Annual demographic updates
    # --------------------------------------------------------

    # First demographic update occurs at t0 + 1.
    # An update at year y advances the population from y-1 to y.

    update_times = collect(
        (t0 + 1.0):1.0:floor(t1)
    )

    function demographic_affect!(integrator)
        transition_year = Int(round(integrator.t)) - 1

        apply_demography!(
            integrator.u,
            model,
            demography,
            transition_year,
        )

        u_modified!(integrator, true)

        return nothing
    end

    callback = PresetTimeCallback(
        update_times,
        demographic_affect!;
        save_positions = (true, true),
    )

    # --------------------------------------------------------
    # ODE problem
    # --------------------------------------------------------

    problem = ODEProblem(
        tb_rhs!,
        copy(u0),
        (t0, t1),
        context,
    )

        # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    return solve(
        problem,
        solver;
        callback = callback,
        save_start = true,
        save_everystep = false,
        save_end = false,
        abstol = abstol,
        reltol = reltol,
    )
end