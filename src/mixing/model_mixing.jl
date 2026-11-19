
"""
    normalise_model_contacts(contact)

Return a validated contact matrix normalised so that
its mean row sum equals one.

No reciprocity correction is performed here.
"""
function normalise_model_contacts(
    contact::AbstractMatrix{<:Real},
)
    n = size(contact, 1)

    n > 0 && size(contact, 2) == n ||
        throw(DimensionMismatch(
            "Contact matrix must be nonempty and square"
        ))

    all(x -> isfinite(x) && x >= 0, contact) ||
        throw(ArgumentError(
            "Contact matrix must be finite and nonnegative"
        ))

    C = Matrix{Float64}(contact)
    mean_contacts = sum(C) / n

    mean_contacts > 0 ||
        throw(ArgumentError(
            "Contact matrix has zero total contacts"
        ))

    return C ./ mean_contacts
end

"""
    adult_infectiousness(nages; minimum_age=15)

Age-specific infectiousness weights, with zero
contribution from children below minimum_age.
"""
function adult_infectiousness(
    nages::Integer;
    minimum_age::Integer=15,
)
    nages > 0 ||
        throw(ArgumentError("nages must be positive"))

    minimum_age >= 0 ||
        throw(ArgumentError(
            "minimum_age must be nonnegative"
        ))

    return Float64[
        age < minimum_age ? 0.0 : 1.0
        for age in 0:(nages-1)
    ]
end
