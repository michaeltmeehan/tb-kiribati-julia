const WPP_COUNTRY_KIRIBATI = "Kiribati"
const WPP_START_YEAR = 1950
const WPP_END_YEAR = 2030
const WPP_YEARS = WPP_START_YEAR:WPP_END_YEAR
const WPP_OPEN_AGE = 95
const WPP_AGE_LABELS = vcat(string.(0:94), "95+")

function _wpp_data_dir()
    return joinpath(@__DIR__, "..", "data", "wpp2024")
end

function _wpp_reference_dir()
    return joinpath(@__DIR__, "..", "validation", "reference", "agepi_kiribati")
end

function _read_simple_csv(path::AbstractString)
    isfile(path) || error("Missing CSV file: $path")
    lines = readlines(path)
    isempty(lines) && error("CSV file is empty: $path")
    headers = String.(split(strip(first(lines)), ","))
    rows = [String.(split(strip(line), ",")) for line in Iterators.drop(lines, 1) if !isempty(strip(line))]
    return headers, rows
end

function _csv_column(headers::AbstractVector{<:AbstractString}, rows::AbstractVector{<:AbstractVector{<:AbstractString}}, name::AbstractString)
    idx = findfirst(==(String(name)), headers)
    idx === nothing && error("CSV file is missing column: $name")
    return [row[idx] for row in rows]
end

function _parse_age_lower(label::AbstractString)
    s = strip(label)
    if isempty(s)
        error("Empty age label")
    end
    if endswith(s, "+")
        return parse(Int, replace(s, "+" => ""))
    end
    if occursin("-", s)
        return parse(Int, split(s, "-"; limit = 2)[1])
    end
    return parse(Int, s)
end

function _age_label(lower::Integer)
    return lower == WPP_OPEN_AGE ? "95+" : string(lower)
end

function _allocate_wpp_matrix(years::AbstractVector{<:Integer} = collect(WPP_YEARS), fill_value::Real = 0.0)
    return fill(Float64(fill_value), NAGE, length(years))
end

function _load_long_table(path::AbstractString)
    headers, rows = _read_simple_csv(path)
    return headers, rows
end

function _parse_numeric_column(headers, rows, name)
    values = _csv_column(headers, rows, name)
    out = Vector{Float64}(undef, length(values))
    @inbounds for i in eachindex(values)
        value = strip(values[i])
        out[i] = isempty(value) || uppercase(value) == "NA" ? 0.0 : parse(Float64, value)
    end
    return out
end

function _parse_int_column(headers, rows, name)
    values = _csv_column(headers, rows, name)
    return parse.(Int, values)
end

function _parse_string_column(headers, rows, name)
    return _csv_column(headers, rows, name)
end

function _extract_year_column(matrix::AbstractMatrix{<:Real}, years::AbstractVector{<:Integer}, year::Integer)
    idx = findfirst(==(Int(year)), years)
    idx === nothing && error("Missing year $year")
    return Float64.(matrix[:, idx])
end

function _maybe_extract_year_column(matrix::AbstractMatrix{<:Real}, years::AbstractVector{<:Integer}, year::Integer)
    idx = findfirst(==(Int(year)), years)
    idx === nothing && return nothing
    return Float64.(matrix[:, idx])
end

function _load_population_reference(path::AbstractString)
    headers, rows = _load_long_table(path)
    ages = _parse_string_column(headers, rows, "age")
    pop = _parse_numeric_column(headers, rows, "population")
    return (; ages, pop)
end

function _load_rate_reference(path::AbstractString, value_col::AbstractString)
    headers, rows = _load_long_table(path)
    years = _parse_int_column(headers, rows, "year")
    ages = _parse_string_column(headers, rows, "age")
    values = _parse_numeric_column(headers, rows, value_col)
    return (; years, ages, values)
end

function _fill_full_age_matrix!(
    matrix::AbstractMatrix{Float64},
    years::Vector{Int},
    ages::Vector{String},
    values::Vector{Float64},
    col_lookup::Dict{Int,Int};
    collapse_open_age::Bool,
    open_age::Int = WPP_OPEN_AGE,
)
    @inbounds for i in eachindex(values)
        year = years[i]
        col = get(col_lookup, year, 0)
        col == 0 && continue
        age_lower = _parse_age_lower(ages[i])
        row = if collapse_open_age && age_lower >= open_age
            open_age + 1
        else
            age_lower + 1
        end
        row < 1 && continue
        row > NAGE && continue
        matrix[row, col] += values[i]
    end
    return matrix
end

function _population_reference_matrix(data_dir::AbstractString)
    ref = _load_population_reference(joinpath(data_dir, "population_2025.csv"))
    pop = zeros(Float64, NAGE)
    for i in eachindex(ref.pop)
        age_lower = _parse_age_lower(ref.ages[i])
        row = if age_lower >= WPP_OPEN_AGE
            NAGE
        else
            age_lower + 1
        end
        pop[row] += ref.pop[i]
    end
    return pop
end

function _build_population_matrix(data_dir::AbstractString, selected_years::AbstractVector{<:Integer})
    headers, rows = _load_long_table(joinpath(data_dir, "population.csv"))
    row_years = _parse_int_column(headers, rows, "year")
    ages = _parse_string_column(headers, rows, "age")
    pop = _parse_numeric_column(headers, rows, "pop")
    matrix = _allocate_wpp_matrix(selected_years)
    col_lookup = Dict(year => i for (i, year) in pairs(collect(selected_years)))
    _fill_full_age_matrix!(matrix, row_years, ages, pop, col_lookup; collapse_open_age = true)
    return matrix
end

function _build_mortality_matrix(data_dir::AbstractString, selected_years::AbstractVector{<:Integer})
    headers, rows = _load_long_table(joinpath(data_dir, "mortality.csv"))
    row_years = _parse_int_column(headers, rows, "year")
    ages = _parse_string_column(headers, rows, "age")
    mx = _parse_numeric_column(headers, rows, "mxB")
    matrix = _allocate_wpp_matrix(selected_years)
    col_lookup = Dict(year => i for (i, year) in pairs(collect(selected_years)))
    @inbounds for i in eachindex(mx)
        year = row_years[i]
        col = get(col_lookup, year, 0)
        col == 0 && continue
        age_lower = _parse_age_lower(ages[i])
        age_lower > WPP_OPEN_AGE && continue
        row = age_lower + 1
        matrix[row, col] = mx[i]
    end
    return matrix
end

function _build_migration_matrix(data_dir::AbstractString, selected_years::AbstractVector{<:Integer})
    headers, rows = _load_long_table(joinpath(data_dir, "migration.csv"))
    row_years = _parse_int_column(headers, rows, "year")
    ages = _parse_string_column(headers, rows, "age")
    mig = _parse_numeric_column(headers, rows, "mig")
    matrix = _allocate_wpp_matrix(selected_years)
    col_lookup = Dict(year => i for (i, year) in pairs(collect(selected_years)))
    _fill_full_age_matrix!(matrix, row_years, ages, 1000.0 .* mig, col_lookup; collapse_open_age = true)
    return matrix
end

function _build_fertility_matrix(data_dir::AbstractString, selected_years::AbstractVector{<:Integer})
    headers, rows = _load_long_table(joinpath(data_dir, "fertility.csv"))
    row_years = _parse_int_column(headers, rows, "year")
    ages = _parse_string_column(headers, rows, "age")
    pasfr = _parse_numeric_column(headers, rows, "pasfr")
    tfr = _parse_numeric_column(headers, rows, "tfr")
    matrix = _allocate_wpp_matrix(selected_years)
    col_lookup = Dict(year => i for (i, year) in pairs(collect(selected_years)))
    @inbounds for i in eachindex(pasfr)
        year = row_years[i]
        col = get(col_lookup, year, 0)
        col == 0 && continue
        age_lower = _parse_age_lower(ages[i])
        row = age_lower + 1
        row < 1 && continue
        row > NAGE && continue
        matrix[row, col] = tfr[i] * (pasfr[i] / 100.0)
    end
    return matrix
end

function _build_births_vector(data_dir::AbstractString, selected_years::AbstractVector{<:Integer})
    headers, rows = _load_long_table(joinpath(data_dir, "births.csv"))
    row_years = _parse_int_column(headers, rows, "year")
    births = _parse_numeric_column(headers, rows, "births")
    vector = zeros(Float64, length(selected_years))
    col_lookup = Dict(year => i for (i, year) in pairs(collect(selected_years)))
    @inbounds for i in eachindex(births)
        col = get(col_lookup, row_years[i], 0)
        col == 0 && continue
        vector[col] = births[i]
    end
    return vector
end

function load_kiribati_wpp_data(; data_dir::AbstractString = _wpp_data_dir(), years::AbstractVector{<:Integer} = collect(WPP_YEARS), fertility_mode::Symbol = :agepi_compatible)
    selected_years = Int[years...]
    population = _build_population_matrix(data_dir, selected_years)
    mortality = _build_mortality_matrix(data_dir, selected_years)
    fertility = _build_fertility_matrix(data_dir, selected_years)
    migration = _build_migration_matrix(data_dir, selected_years)
    births = _build_births_vector(data_dir, selected_years)
    schedule = DemographicSchedule(selected_years, mortality, fertility, migration; births = births, fertility_mode = fertility_mode)
    population_1950 = _maybe_extract_year_column(population, selected_years, WPP_START_YEAR)
    population_2025 = _maybe_extract_year_column(population, selected_years, 2025)
    return (
        country = WPP_COUNTRY_KIRIBATI,
        years = selected_years,
        population_1950 = population_1950,
        population_2025 = population_2025,
        population = population,
        mortality = mortality,
        fertility = fertility,
        migration = migration,
        births = births,
        schedule = schedule,
    )
end

function load_kiribati_agepi_reference(; reference_dir::AbstractString = _wpp_reference_dir())
    population = _population_reference_matrix(reference_dir)
    headers_m, rows_m = _load_long_table(joinpath(reference_dir, "mortality_2025_2030.csv"))
    headers_f, rows_f = _load_long_table(joinpath(reference_dir, "fertility_2025_2030.csv"))
    headers_g, rows_g = _load_long_table(joinpath(reference_dir, "migration_2025_2030.csv"))
    years_m = _parse_int_column(headers_m, rows_m, "time")
    ages_m = _parse_string_column(headers_m, rows_m, "age_group")
    vals_m = _parse_numeric_column(headers_m, rows_m, "mortality_rate")
    years_f = _parse_int_column(headers_f, rows_f, "time")
    ages_f = _parse_string_column(headers_f, rows_f, "age_group")
    vals_f = _parse_numeric_column(headers_f, rows_f, "fertility_rate")
    years_g = _parse_int_column(headers_g, rows_g, "time")
    ages_g = _parse_string_column(headers_g, rows_g, "age_group")
    vals_g = _parse_numeric_column(headers_g, rows_g, "migration_count")
    mortality = _allocate_wpp_matrix()
    fertility = _allocate_wpp_matrix()
    migration = _allocate_wpp_matrix()
    col_lookup = Dict(year => i for (i, year) in pairs(collect(WPP_YEARS)))
    _fill_full_age_matrix!(mortality, years_m, ages_m, vals_m, col_lookup; collapse_open_age = true)
    _fill_full_age_matrix!(fertility, years_f, ages_f, vals_f, col_lookup; collapse_open_age = false)
    _fill_full_age_matrix!(migration, years_g, ages_g, vals_g, col_lookup; collapse_open_age = true)
    return (; population, mortality, fertility, migration)
end

function _reference_population_vector(population::AbstractVector{<:Real})
    length(population) == NAGE || error("population must have length 96")
    return Float64.(population)
end

function wpp_kiribati_population(year::Integer = 2025; data_dir::AbstractString = _wpp_data_dir())
    data = load_kiribati_wpp_data(data_dir = data_dir)
    return _reference_population_vector(_extract_year_column(data.population, data.years, year))
end

function wpp_kiribati_population_2025(; data_dir::AbstractString = _wpp_data_dir())
    return wpp_kiribati_population(2025; data_dir = data_dir)
end

function wpp_kiribati_population_1950(; data_dir::AbstractString = _wpp_data_dir())
    return wpp_kiribati_population(1950; data_dir = data_dir)
end

function wpp_kiribati_demographic_schedule(; data_dir::AbstractString = _wpp_data_dir(), years::AbstractVector{<:Integer} = collect(WPP_YEARS), fertility_mode::Symbol = :agepi_compatible)
    return load_kiribati_wpp_data(data_dir = data_dir, years = years, fertility_mode = fertility_mode).schedule
end
