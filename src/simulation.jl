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
    demography = :static,
    annual_update_times = nothing,
    solver = Vern7(),
    kwargs...,
)
    if isnothing(annual_update_times)
        start_year = ceil(Int, first(tspan)) + 1
        end_year = floor(Int, last(tspan))
        annual_update_times = collect(start_year:end_year)
    end

    callback = make_annual_callback(
        annual_update_times;
        demography = demography,
    )

    prob = ODEProblem(
        tb_rhs!,
        u0,
        tspan,
        params,
    )

    return solve(
        prob,
        solver;
        saveat = saveat,
        callback = callback,
        kwargs...,
    )
end