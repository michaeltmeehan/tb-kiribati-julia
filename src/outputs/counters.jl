"""
    annual_events(sol, layout::StateLayout, counter::Integer; years)

Return an `nages × length(years)` matrix of annual event counts from a cumulative
counter. Entry `[a,j]` counts events in age class `a-1` during `[years[j], years[j]+1)`.

At a demographic discontinuity, take the *last* saved state at the interval's
start (t⁺), and the *first* saved state at its end ((t+1)⁻). The simulation
must save both states at each demographic update (`save_positions=(true,true)`).
"""
function annual_events(sol, layout::StateLayout, counter::Integer; years)
    layout.nepi < counter <= nstates(layout) ||
        throw(ArgumentError("counter must index a cumulative event state"))

    yy = collect(years)
    all(y -> y isa Integer, yy) ||
        throw(ArgumentError("years must contain integer calendar years"))
    length(unique(yy)) == length(yy) ||
        throw(ArgumentError("years must not contain duplicates"))

    length(sol.t) == length(sol.u) ||
        throw(DimensionMismatch("solution times and states differ in length"))
    isempty(sol.t) && throw(ArgumentError("solution has no saved states"))
    all(u -> length(u) == layout.nages * nstates(layout), sol.u) ||
        throw(DimensionMismatch("saved state length does not match layout"))

    out = Matrix{Float64}(undef, layout.nages, length(yy))
    tol = 1e-7

    for (j, y) in enumerate(yy)
        # Exact calendar boundaries (within solver floating-point tolerance).
        start_indices = findall(t -> abs(t - y) <= tol, sol.t)
        end_indices = findall(t -> abs(t - (y + 1)) <= tol, sol.t)
        isempty(start_indices) &&
            throw(ArgumentError("no saved state at start of year $y"))
        isempty(end_indices) &&
            throw(ArgumentError("no saved state at end of year $y"))

        # Interior boundaries must have BOTH sides saved.
        if y > first(sol.t) + tol && length(start_indices) < 2
            throw(ArgumentError("no pre/post demographic pair at year $y"))
        end
        if y + 1 < last(sol.t) - tol && length(end_indices) < 2
            throw(ArgumentError("no pre/post demographic pair at year $(y+1)"))
        end

        before = sol.u[last(start_indices)]  # y⁺
        after  = sol.u[first(end_indices)]   # (y+1)⁻
        for a in 1:layout.nages
            i = state_index(layout, a, Int(counter))
            diff = Float64(after[i] - before[i])
            isfinite(diff) ||
                throw(ArgumentError("nonfinite event count in age $a, year $y"))
            # Permit negligible solver roundoff, but reject real counter decreases.
            diff >= -1e-8 * max(1.0, abs(before[i]), abs(after[i])) ||
                throw(ArgumentError("cumulative counter decreased in age $a, year $y"))
            out[a, j] = max(0.0, diff)
        end
    end
    return out
end

annual_events(sol, model::AbstractTBModel, counter::Integer; years) =
    annual_events(sol, state_layout(model), counter; years)