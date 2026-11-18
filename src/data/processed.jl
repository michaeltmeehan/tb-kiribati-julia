
module ProcessedData

using CSV
using DataFrames

export load_surveillance, load_estimates,
    load_demography, load_contacts

const DEFAULT_ROOT = normpath(
    joinpath(@__DIR__, "..", "..")
)

const SURVEILLANCE = Dict(
    :notifications_total => "tb_notifications_total",
    :notifications_by_age => "tb_notifications_by_age",
    :notifications_by_site => "tb_notifications_by_site",
    :recurrent => "tb_recurrent_notifications",
    :treatment_outcomes => "tb_treatment_outcomes",
    :treatment_main => "tb_treatment_outcomes_main",
    :treatment_age_sex => "tb_treatment_outcomes_age_sex",
    :historical_presentation => "tb_historical_new_notifications_by_age_and_type",
    :historical_site_age => "tb_new_notifications_by_site_age_2007_2012",
    :historical_pulmonary_fraction => "tb_pulmonary_fraction_by_age_2007_2012",
)

const ESTIMATES = Dict(
    :burden_national => "tb_burden_national",
    :incidence_age_sex => "tb_incidence_age_sex",
    :preventive_treatment => "tb_preventive_treatment",
)

const DEMOGRAPHY = Dict(
    :population => "population",
    :fertility => "fertility",
    :mortality => "mortality",
    :migration => "migration",
    :vital_totals => "vital_totals",
    :population_totals => "population_totals",
    :fertility_checks => "fertility_checks",
)

function read_processed(
    category::AbstractString,
    dataset::Symbol,
    registry::Dict;
    root::AbstractString=DEFAULT_ROOT,
    years=nothing,
)
    haskey(registry, dataset) ||
        throw(ArgumentError(
            "Unknown dataset $dataset. Available: " *
                join(sort!(collect(keys(registry))), ", ")
        ))

    path = joinpath(
        root, "data", "processed",
        category, registry[dataset] * ".csv"
    )

    isfile(path) ||
        throw(ArgumentError("Dataset not found: $path"))

    df = CSV.read(path, DataFrame)

    if !isnothing(years)
        :year in propertynames(df) ||
            throw(ArgumentError(
                "Dataset $dataset has no year column"
            ))

        selected = Set(Int.(collect(years)))
        filter!(:year => in(selected), df)
    end

    return df
end

"""
    load_surveillance(dataset; years=nothing, root=DEFAULT_ROOT)

Load reported WHO 2026 surveillance observations.
"""
function load_surveillance(
    dataset::Symbol;
    years=nothing,
    root::AbstractString=DEFAULT_ROOT,
)
    return read_processed(
        "surveillance", dataset, SURVEILLANCE;
        years, root
    )
end

"""
    load_estimates(dataset; years=nothing, root=DEFAULT_ROOT)

Load WHO-generated burden or preventive-treatment estimates.
"""
function load_estimates(
    dataset::Symbol;
    years=nothing,
    root::AbstractString=DEFAULT_ROOT,
)
    return read_processed(
        joinpath("surveillance", "estimates"),
        dataset, ESTIMATES;
        years, root
    )
end

"""
    load_demography(dataset; years=nothing, root=DEFAULT_ROOT)

Load processed WPP2024 demographic inputs.
Population is expressed in persons.
"""
function load_demography(
    dataset::Symbol;
    years=nothing,
    root::AbstractString=DEFAULT_ROOT,
)
    return read_processed(
        "demography", dataset, DEMOGRAPHY;
        years, root
    )
end

"""
    load_contacts(; variant=:original, root=DEFAULT_ROOT)

Load an unnormalised Prem contact matrix.

Available variants:
- :original         96×96 original expanded matrix
- :reciprocal_2023  96×96 Kiribati-adjusted matrix

Rows represent recipient ages; columns represent source ages.
"""
function load_contacts(;
    variant::Symbol=:original,
    root::AbstractString=DEFAULT_ROOT,
)
    filename = if variant == :original
        "contact_matrix.csv"
    elseif variant == :reciprocal_2023
        "prem_reciprocal_2023_96.csv"
    else
        throw(ArgumentError(
            "Unknown contact variant: $variant"
        ))
    end

    path = joinpath(
        root, "data", "processed", "contacts", filename
    )

    isfile(path) ||
        throw(ArgumentError("Contact file not found: $path"))

    df = CSV.read(path, DataFrame)

    nrow(df) == 96 ||
        throw(DimensionMismatch("Expected 96 recipient ages"))

    df.recipient_age == collect(0:95) ||
        throw(ArgumentError("Invalid recipient age ordering"))

    expected = ["source_age_$a" for a in 0:95]

    names(df)[2:end] == expected ||
        throw(ArgumentError("Invalid source age ordering"))

    C = Matrix{Float64}(df[:, 2:end])

    all(x -> isfinite(x) && x >= 0, C) ||
        throw(ArgumentError("Invalid contact matrix"))

    return C
end

end # module
