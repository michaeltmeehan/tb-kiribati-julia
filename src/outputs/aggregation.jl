
"""
Aggregate single-year age outputs into age groups.

Input and output have dimensions age × time.
The input numerical type is preserved.
"""
function aggregate_age_groups(
    values::AbstractMatrix{T},
    age_breaks::AbstractVector,
) where {T<:Real}
    nages, ntimes = size(values)
    breaks = collect(age_breaks)

    length(breaks) >= 2 ||
        throw(ArgumentError("At least two breaks required"))

    all(x -> x isa Integer, breaks) ||
        throw(ArgumentError("Age breaks must be integers"))

    first(breaks) == 0 ||
        throw(ArgumentError("First age break must be zero"))

    last(breaks) == nages ||
        throw(ArgumentError(
            "Last age break must equal number of ages"
        ))

    all(diff(breaks) .> 0) ||
        throw(ArgumentError(
            "Age breaks must be strictly increasing"
        ))

    result = zeros(T, length(breaks) - 1, ntimes)

    for g in 1:(length(breaks)-1)
        for a in (breaks[g]+1):breaks[g+1]
            for t in 1:ntimes
                result[g, t] += values[a, t]
            end
        end
    end

    return result
end
