
"""
Saved solution indices for annual event extraction.

For year y:
- start_index refers to y⁺
- end_index refers to (y+1)⁻
"""
struct AnnualWindows
    years::Vector{Int}
    start_indices::Vector{Int}
    end_indices::Vector{Int}
end

function annual_windows(sol; years)
    raw_years = collect(years)

    all(y -> y isa Integer, raw_years) ||
        throw(ArgumentError("Years must be integers"))

    yy = Int.(raw_years)

    length(unique(yy)) == length(yy) ||
        throw(ArgumentError("Duplicate years"))

    length(sol.t) == length(sol.u) ||
        throw(DimensionMismatch(
            "Solution times and states differ in length"
        ))

    isempty(sol.t) &&
        throw(ArgumentError("Solution contains no saved states"))

    tol = 1e-7
    starts = Int[]
    ends = Int[]

    for y in yy

        si = findall(t -> abs(t - y) <= tol, sol.t)
        ei = findall(t -> abs(t - (y + 1)) <= tol, sol.t)

        isempty(si) &&
            throw(ArgumentError("Missing start of year $y"))

        isempty(ei) &&
            throw(ArgumentError("Missing end of year $y"))

        if y > first(sol.t) + tol && length(si) < 2
            throw(ArgumentError(
                "Missing demographic boundary pair at $y"
            ))
        end

        if y + 1 < last(sol.t) - tol && length(ei) < 2
            throw(ArgumentError(
                "Missing demographic boundary pair at $(y+1)"
            ))
        end

        push!(starts, last(si))
        push!(ends, first(ei))
    end

    return AnnualWindows(yy, starts, ends)
end


"""
Extract annual increments of a cumulative event counter.

A small negative increment caused by numerical error is
permitted, but is not clipped. Larger decreases are rejected.

This avoids introducing a nondifferentiable max operation
into the inference-facing extraction.
"""
function annual_events(
    sol,
    layout::StateLayout,
    counter::Integer,
    windows::AnnualWindows,
)
    layout.nepi < counter <= nstates(layout) ||
        throw(ArgumentError("Invalid cumulative counter"))

    expected_length = state_dimension(layout)

    all(u -> length(u) == expected_length, sol.u) ||
        throw(DimensionMismatch(
            "Solution state dimensions do not match layout"
        ))

    T = eltype(first(sol.u))

    out = Matrix{T}(
        undef,
        layout.nages,
        length(windows.years),
    )

    for j in eachindex(windows.years)
        before = sol.u[windows.start_indices[j]]
        after = sol.u[windows.end_indices[j]]

        for a in 1:layout.nages
            i = state_index(layout, a, Int(counter))
            diff = after[i] - before[i]

            isfinite(diff) ||
                throw(ArgumentError(
                    "Nonfinite event increment"
                ))

            tolerance = 1e-8 * max(
                one(diff),
                abs(before[i]),
                abs(after[i]),
            )

            diff >= -tolerance ||
                throw(ArgumentError(
                    "Cumulative counter decreased"
                ))

            out[a, j] = diff
        end
    end

    return out
end


function annual_events(
    sol,
    layout::StateLayout,
    counter::Integer;
    years,
)
    windows = annual_windows(sol; years)
    return annual_events(sol, layout, counter, windows)
end


annual_events(
    sol,
    model::AbstractTBModel,
    counter::Integer;
    years,
) = annual_events(
    sol, state_layout(model), counter; years
)
