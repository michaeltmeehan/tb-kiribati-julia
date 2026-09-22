import DiffEqCallbacks as CB
using DifferentialEquations: ODEProblem, solve
using OrdinaryDiffEq: Vern7


function reset_cumulative_counters!(u)
    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE

        for c in (NEPI + 1):NSTATE
            u[base + c] = 0.0
        end
    end

    return nothing
end


function make_annual_callback(times; demography = :static)

    demography in (:static, :dynamic, :none) ||
        error("demography must be :static, :dynamic, or :none")

    function annual_update!(integrator)

        if demography === :static
            _apply_static_demography!(integrator.u)

        elseif demography === :dynamic
            apply_demography!(integrator)
        end

        reset_cumulative_counters!(integrator.u)

        return nothing
    end

    return CB.PresetTimeCallback(times, annual_update!)
end


function simulate(
    params;
    tspan,
    u0,
    saveat = 1.0,
    callback = nothing,
    solver = Vern7(),
    kwargs...,
)
    prob = ODEProblem(
        tb_rhs!,
        u0,
        tspan,
        params,
    )

    if isnothing(callback)
        return solve(
            prob,
            solver;
            saveat = saveat,
            kwargs...,
        )
    end

    return solve(
        prob,
        solver;
        saveat = saveat,
        callback = callback,
        kwargs...,
    )
end