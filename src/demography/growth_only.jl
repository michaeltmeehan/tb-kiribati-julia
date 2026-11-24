struct GrowthOnlyDemography <: AbstractDemography
    data::DemographicData
    reference_year::Int
    age_distribution::Vector{Float64}
end

function GrowthOnlyDemography(
    data::DemographicData,
    reference_year::Integer;
    nages::Int=96,
)
    population = get_population(
        data, reference_year; nages
    )

    total = sum(population)

    total > 0 ||
        throw(ArgumentError("Reference population must be positive"))

    return GrowthOnlyDemography(
        data,
        Int(reference_year),
        population ./ total,
    )
end

function get_population(
    demography::GrowthOnlyDemography,
    year::Real,
)
    population = get_population(
        demography.data,
        year;
        nages=length(demography.age_distribution),
    )

    return sum(population) .* demography.age_distribution
end

function target_population(
    demography::GrowthOnlyDemography,
    year::Real,
)
    return get_population(demography, year + 1)
end

function demographic_mortality_year(
    demography::GrowthOnlyDemography,
    year::Real,
)
    return demography.reference_year
end