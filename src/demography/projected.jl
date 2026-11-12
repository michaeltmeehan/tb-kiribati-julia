# src/demography/projected.jl

"""
    ProjectedDemography(data; nages=96)

Follow the supplied annual age-specific population projections.
"""
struct ProjectedDemography <: AbstractDemography
    data::DemographicData
    nages::Int
end


function ProjectedDemography(
    data::DemographicData;
    nages::Int=96,
)
    2 <= nages <= size(data.population, 1) ||
        throw(ArgumentError("Invalid number of age classes"))

    return ProjectedDemography(data, nages)
end


"""
Return the projected population for the year following
the demographic update.
"""
function target_population(
    demography::ProjectedDemography,
    year::Real,
)
    return get_population(
        demography.data,
        year + 1;
        nages=demography.nages,
    )
end


function get_population(
    demography::ProjectedDemography,
    year::Real,
)
    return get_population(
        demography.data,
        year;
        nages=demography.nages,
    )
end