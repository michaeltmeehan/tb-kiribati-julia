# src/demography/equilibrium.jl

"""
    EquilibriumDemography(data, reference_year; nages=96)

Maintain a fixed age-specific population corresponding to
the specified reference year.
"""
struct EquilibriumDemography <: AbstractDemography
    data::DemographicData
    reference_year::Int
    target_population::Vector{Float64}
end


function EquilibriumDemography(
    data::DemographicData,
    reference_year::Integer;
    nages::Int=96,
)
    target = get_population(
        data,
        reference_year;
        nages=nages,
    )

    return EquilibriumDemography(
        data,
        Int(reference_year),
        target,
    )
end


"""
Return the fixed reference population, irrespective of year.
"""
target_population(
    demography::EquilibriumDemography,
    year::Real,
) = demography.target_population


function get_population(
    demography::EquilibriumDemography,
    year::Real,
)
    return copy(demography.target_population)
end