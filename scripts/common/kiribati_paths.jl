"""
Resolve processed data directories and legacy numerical inputs.
"""
function kiribati_data_paths(root::AbstractString)
    data = joinpath(root, "data")
    processed = joinpath(data, "processed")

    return (
        root=root,
        data=data,
        surveillance=joinpath(processed, "surveillance"),
        estimates=joinpath(processed, "surveillance", "estimates"),
        demography=joinpath(processed, "demography"),
        contacts=joinpath(processed, "contacts"),
        legacy_demography=data,
        legacy_contact=joinpath(data, "contact"),
    )
end