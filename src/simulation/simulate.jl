
using OrdinaryDiffEq
using DiffEqCallbacks: PresetTimeCallback
using SciMLBase: ODEProblem, solve, remake, u_modified!

# ------------------------------------------------------------
# Numerical type inference
# ------------------------------------------------------------

function simulation_parameter_type(x)
    if x isa Real
        return typeof(x)
    elseif x isa AbstractArray{<:Real}
        return eltype(x)
    elseif x isa Tuple
        return foldl(
            promote_type,
            (simulation_parameter_type(v) for v in x);
            init=Float64,
        )
    elseif isstructtype(typeof(x)) &&
        !(x isa AbstractString)
        return foldl(
            promote_type,
            (
                simulation_parameter_type(getfield(x, i))
                for i in 1:fieldcount(typeof(x))
            );
            init=Float64,
        )
    else
        return Float64
    end
end


# ------------------------------------------------------------
# Prepared simulation
# ------------------------------------------------------------

"""
    PreparedSimulation

Reusable deterministic simulation configuration.

Contains:
- Fixed model and demographic scenario
- ODE problem template
- Annual demographic callback schedule
- Solver and numerical tolerances
- Output-saving configuration

Individual solves receive fresh parameters, initial states,
contexts and workspaces.
"""
struct PreparedSimulation{M,D,P,C,S,A,R,V}
    model::M
    demography::D
    problem::P
    callback::C
    solver::S
    abstol::A
    reltol::R
    saveat::V
    save_end::Bool
end


"""
    prepare_simulation(model, demography;
                       start_year, end_year, ...)

Prepare an ODE problem for repeated parameter evaluations.

The callback reads the current integrator context rather
than capturing any proposal-specific parameter values.
"""
function prepare_simulation(
    model::AbstractTBModel,
    demography::AbstractDemography;
    start_year::Real,
    end_year::Real,
    solver=Tsit5(),
    abstol::Real=1e-8,
    reltol::Real=1e-6,
    saveat=nothing,
)
    t0 = Float64(start_year)
    t1 = Float64(end_year)

    isfinite(t0) && isfinite(t1) && t0 < t1 ||
        throw(ArgumentError("Invalid simulation interval"))

    isinteger(t0) ||
        throw(ArgumentError(
            "Simulation must begin at an integer calendar year"
        ))

    layout = state_layout(model)

    # Placeholder values are never integrated directly.
    # remake substitutes the actual state and context.
    prototype_context = SimulationContext(
        model,
        nothing,
        demography,
        TBWorkspace(layout.nages),
    )

    problem = ODEProblem(
        tb_rhs!,
        zeros(Float64, state_dimension(layout)),
        (t0, t1),
        prototype_context,
    )

    update_times = collect(
        (t0+1.0):1.0:floor(t1)
    )

    function demographic_affect!(integrator)
        ctx = integrator.p
        previous_year = Int(round(integrator.t)) - 1

        apply_demography!(
            integrator.u,
            ctx.model,
            ctx.demography,
            previous_year,
        )

        u_modified!(integrator, true)
        return nothing
    end

    callback = isempty(update_times) ?
               nothing :
               PresetTimeCallback(
        update_times,
        demographic_affect!;
        save_positions=(true, true),
    )

    save_end = isempty(update_times) || !isinteger(t1)

    return PreparedSimulation(
        model,
        demography,
        problem,
        callback,
        solver,
        abstol,
        reltol,
        saveat,
        save_end,
    )
end


# ------------------------------------------------------------
# Repeated parameter evaluation
# ------------------------------------------------------------

"""
    simulate(setup::PreparedSimulation, params; u0=nothing)

Run a prepared simulation with new model parameters.

If u0 is omitted, initial_state is dispatched on the model.
No model reconstruction occurs.

A fresh workspace is allocated for every solve, avoiding
shared mutable state between parameter evaluations.
"""
function simulate(
    setup::PreparedSimulation,
    params;
    u0=nothing,
)
    layout = state_layout(setup.model)

    initial = isnothing(u0) ?
              initial_state(
        setup.model,
        params,
        setup.demography;
        year=setup.problem.tspan[1],
    ) :
              u0

    initial isa AbstractVector{<:Real} ||
        throw(ArgumentError(
            "Initial state must be a real-valued vector"
        ))

    length(initial) == state_dimension(layout) ||
        throw(DimensionMismatch(
            "Initial state has incorrect dimension"
        ))

    all(isfinite, initial) ||
        throw(ArgumentError(
            "Initial state contains nonfinite values"
        ))

    T = promote_type(
        eltype(initial),
        simulation_parameter_type(params),
    )

    context = SimulationContext(
        setup.model,
        params,
        setup.demography,
        TBWorkspace(T, layout.nages),
    )

    problem = remake(
        setup.problem;
        u0=T.(initial),
        p=context,
    )

    options = (
        save_start=true,
        save_everystep=false,
        save_end=setup.save_end,
        abstol=setup.abstol,
        reltol=setup.reltol,
    )

    if !isnothing(setup.callback)
        options = merge(
            options,
            (callback=setup.callback,),
        )
    end

    if !isnothing(setup.saveat)
        options = merge(
            options,
            (saveat=setup.saveat,),
        )
    end

    return solve(
        problem,
        setup.solver;
        options...,
    )
end


"""
    simulate_from_draw(setup::PreparedSimulation, draw)

Construct parameters and solve using a prepared simulation.
"""
function simulate_from_draw(
    setup::PreparedSimulation,
    draw::NamedTuple,
)
    params = make_parameters(setup.model, draw)
    return simulate(setup, params)
end


# ------------------------------------------------------------
# Backward-compatible convenience interfaces
# ------------------------------------------------------------

function simulate(
    model::AbstractTBModel,
    params,
    demography::AbstractDemography;
    u0::AbstractVector,
    tspan::Tuple{<:Real,<:Real},
    kwargs...,
)
    setup = prepare_simulation(
        model,
        demography;
        start_year=tspan[1],
        end_year=tspan[2],
        kwargs...,
    )

    return simulate(setup, params; u0=u0)
end


function simulate_from_draw(
    model::AbstractTBModel,
    draw::NamedTuple,
    demography::AbstractDemography;
    start_year::Real,
    end_year::Real,
    kwargs...,
)
    setup = prepare_simulation(
        model,
        demography;
        start_year,
        end_year,
        kwargs...,
    )

    return simulate_from_draw(setup, draw)
end
