import DiffEqCallbacks as CB


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