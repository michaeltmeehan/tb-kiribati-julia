# src/data/contacts.jl

using CSV
using DataFrames
using DelimitedFiles: readdlm

"""
    load_contact_matrix(path; nages=nothing)

Load a whitespace-delimited contact matrix.

Returns a Matrix{Float64}, validating that it is square,
finite, nonnegative, and optionally of the expected size.
"""
function load_contact_matrix(
    path::AbstractString;
    nages::Union{Nothing,Int} = nothing,
)
    isfile(path) ||
        throw(ArgumentError("Contact matrix file not found: $path"))

    contact = Matrix{Float64}(readdlm(path, Float64))

    validate_contact_matrix(contact; nages=nages)

    return contact
end


"""
    validate_contact_matrix(matrix; nages=nothing)

Validate the dimensions and entries of a contact matrix.
"""
function validate_contact_matrix(
    matrix::AbstractMatrix;
    nages::Union{Nothing,Int} = nothing,
)
    nrows, ncols = size(matrix)

    nrows == ncols ||
        throw(DimensionMismatch(
            "Contact matrix must be square"
        ))

    nrows > 0 ||
        throw(ArgumentError(
            "Contact matrix must not be empty"
        ))

    if !isnothing(nages)
        nrows == nages ||
            throw(DimensionMismatch(
                "Expected a $nages × $nages contact matrix"
            ))
    end

    all(x -> isfinite(x) && x >= 0, matrix) ||
        throw(ArgumentError(
            "Contact matrix must contain finite, nonnegative values"
        ))

    return nothing
end


"""
    aggregate_contact_matrix(
        contact,
        population,
        age_breaks,
    )

Aggregate a single-year-age contact matrix into broader
age groups using population-weighted row averages.

`age_breaks` specifies zero-based age boundaries.

For example:
    [0, 5, 15, 25, 96]

produces age groups:
    0–4, 5–14, 15–24, 25+.

Rows are weighted by the population of the corresponding
recipient age. Columns are summed across source ages.
"""
function aggregate_contact_matrix(
    contact::AbstractMatrix,
    population::AbstractVector,
    age_breaks::AbstractVector{<:Integer},
)
    validate_contact_matrix(contact)

    nages = size(contact, 1)

    length(population) == nages ||
        throw(DimensionMismatch(
            "Population length does not match contact matrix"
        ))

    length(age_breaks) >= 2 ||
        throw(ArgumentError(
            "At least two age boundaries are required"
        ))

    first(age_breaks) == 0 ||
        throw(ArgumentError(
            "Age boundaries must begin at zero"
        ))

    last(age_breaks) == nages ||
        throw(ArgumentError(
            "Final age boundary must equal $nages"
        ))

    all(diff(age_breaks) .> 0) ||
        throw(ArgumentError(
            "Age boundaries must be strictly increasing"
        ))

    all(x -> isfinite(x) && x >= 0, population) ||
        throw(ArgumentError(
            "Population must be finite and nonnegative"
        ))

    ngroups = length(age_breaks) - 1
    aggregated = zeros(Float64, ngroups, ngroups)

    for i in 1:ngroups
        rows = (age_breaks[i] + 1):age_breaks[i + 1]

        group_population = sum(@view population[rows])

        group_population > 0 ||
            throw(ArgumentError(
                "Age group $i has zero population"
            ))

        for j in 1:ngroups
            cols = (age_breaks[j] + 1):age_breaks[j + 1]

            total = 0.0

            for a in rows
                contacts = sum(@view contact[a, cols])

                total += population[a] * contacts
            end

            aggregated[i, j] = total / group_population
        end
    end

    return aggregated
end