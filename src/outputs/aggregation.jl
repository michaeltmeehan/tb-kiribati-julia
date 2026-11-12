"""
    aggregate_age_groups(values, age_breaks)

Aggregate a matrix whose rows are single-year ages 0,1,... and whose columns
are years. `age_breaks` consists of integer boundaries such as
`[0,5,15,25,35,45,55,65,96]`. Every single-year age must be covered.
The last group may include the model's open-ended final age class.
"""
function aggregate_age_groups(values::AbstractMatrix, age_breaks::AbstractVector)
    nages = size(values, 1)
    breaks = collect(age_breaks)
    length(breaks) >= 2 || throw(ArgumentError("at least two age breaks required"))
    all(x -> x isa Integer, breaks) ||
        throw(ArgumentError("age breaks must be integers"))
    first(breaks) == 0 || throw(ArgumentError("first age break must be 0"))
    last(breaks) == nages ||
        throw(ArgumentError("last age break must equal number of single-year age classes"))
    all(diff(breaks) .> 0) ||
        throw(ArgumentError("age breaks must be strictly increasing"))
    result = zeros(Float64, length(breaks) - 1, size(values, 2))
    for g in 1:(length(breaks) - 1)
        for a in (breaks[g] + 1):breaks[g + 1]
            @views result[g, :] .+= values[a, :]
        end
    end
    return result
end