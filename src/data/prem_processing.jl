
module PremProcessing

using DataFrames

export expand_contacts, collapse_contacts,
    contact_dataframe

# Prem groups: 0-4, ..., 70-74, 75+
age_band(age::Integer) = min(div(age, 5) + 1, 16)

function expand_contacts(
    source::AbstractMatrix;
    nages::Int=96,
)
    size(source) == (16, 16) ||
        throw(DimensionMismatch("Expected 16×16 matrix"))

    76 <= nages <= 101 ||
        throw(ArgumentError("Invalid number of ages"))

    all(x -> isfinite(x) && x >= 0, source) ||
        throw(ArgumentError("Invalid contact values"))

    groups = [age_band(a) for a in 0:(nages-1)]
    sizes = [count(==(g), groups) for g in 1:16]

    return [
        source[groups[i], groups[j]] /
            sizes[groups[j]]
        for i in 1:nages, j in 1:nages
    ]
end

# Recover the original band-level contact totals.
function collapse_contacts(
    expanded::AbstractMatrix;
)
    n = size(expanded, 1)
    size(expanded) == (n, n) ||
        throw(DimensionMismatch("Contact matrix must be square"))

    groups = [age_band(a) for a in 0:(n-1)]

    return [
        begin
            recipient = findfirst(==(i), groups)
            sources = findall(==(j), groups)

            sum(expanded[recipient, sources])
        end
        for i in 1:16, j in 1:16
    ]
end

function contact_dataframe(C::AbstractMatrix)
    n = size(C, 1)
    size(C) == (n, n) ||
        throw(DimensionMismatch("Contact matrix must be square"))

    df = DataFrame(recipient_age=collect(0:(n-1)))

    for age in 0:(n-1)
        df[!, Symbol("source_age_$age")] = C[:, age+1]
    end

    return df
end

end # module
