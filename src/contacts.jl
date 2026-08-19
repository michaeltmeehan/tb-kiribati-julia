using DelimitedFiles: readdlm

struct ContactAgeBand
    label::String
    lower::Int
    upper::Float64
end

const CONTACT_DATA_DIR = joinpath(@__DIR__, "..", "data", "contact")

const SOLI_CONTACT_SOURCE_FILE = joinpath(CONTACT_DATA_DIR, "prem_solomon_islands_all_source_matrix.csv")
const SOLI_CONTACT_BANDS_FILE = joinpath(CONTACT_DATA_DIR, "prem_solomon_islands_all_agebands.csv")

const SOLI_CONTACT_SOURCE_CACHE = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
const SOLI_CONTACT_FULL_CACHE = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
const SOLI_CONTACT_SOURCE_DATA_CACHE = Ref{Union{Nothing, NamedTuple}}(nothing)

function _read_numeric_matrix(path::AbstractString)
    isfile(path) || error("missing contact matrix file: $path")
    matrix = Matrix{Float64}(readdlm(path, ',', Float64))
    size(matrix, 1) == size(matrix, 2) || error("contact matrix must be square: $path")
    return matrix
end

function _read_contact_age_bands(path::AbstractString)
    isfile(path) || error("missing contact age-band file: $path")

    bands = ContactAgeBand[]
    open(path, "r") do io
        header_seen = false
        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            fields = split(stripped, ',')
            if !header_seen
                header_seen = true
                if !isempty(fields) && lowercase(strip(fields[1])) == "label"
                    continue
                end
            end
            if isempty(fields) || lowercase(strip(fields[1])) == "label"
                continue
            end

            length(fields) ≥ 3 || error("invalid contact age-band row: $line")
            label = strip(fields[1])
            lower = parse(Int, strip(fields[2]))
            upper_text = strip(fields[3])
            upper = isempty(upper_text) ? Inf : parse(Float64, upper_text)
            push!(bands, ContactAgeBand(label, lower, upper))
        end
    end

    isempty(bands) && error("contact age-band file has no data: $path")
    return bands
end

function _validate_contact_age_bands(source_age_bands::AbstractVector{<:ContactAgeBand})
    isempty(source_age_bands) && error("source_age_bands must not be empty")
    @inbounds for i in eachindex(source_age_bands)
        band = source_age_bands[i]
        if i > 1 && source_age_bands[i - 1].lower ≥ band.lower
            error("source age bands must be sorted by increasing lower bound")
        end
        band.lower ≥ 0 || error("source age bands must start at age 0 or above")
        band.upper ≥ band.lower || error("source age bands must have upper >= lower")
    end
    return nothing
end

function _source_band_index(age::Int, source_age_bands::AbstractVector{<:ContactAgeBand})
    @inbounds for i in eachindex(source_age_bands)
        band = source_age_bands[i]
        upper = isfinite(band.upper) ? Int(band.upper) : typemax(Int)
        if band.lower ≤ age ≤ upper
            return i
        end
    end
    error("age $age is outside the source age bands")
end

function _source_band_target_indices(band::ContactAgeBand, max_age::Int)
    upper = isfinite(band.upper) ? min(Int(band.upper), max_age) : max_age
    band.lower ≤ upper || error("source age band $(band.label) is not compatible with max_age=$max_age")
    return collect((band.lower + 1):(upper + 1))
end

function load_solomon_islands_contact_source(; data_dir::AbstractString = CONTACT_DATA_DIR)
    if data_dir == CONTACT_DATA_DIR
        cached = SOLI_CONTACT_SOURCE_DATA_CACHE[]
        if cached !== nothing
            return cached
        end
    end

    matrix = _read_numeric_matrix(joinpath(data_dir, "prem_solomon_islands_all_source_matrix.csv"))
    age_bands = _read_contact_age_bands(joinpath(data_dir, "prem_solomon_islands_all_agebands.csv"))
    _validate_contact_age_bands(age_bands)
    size(matrix, 1) == length(age_bands) || error("source matrix rows must match the number of source age bands")
    size(matrix, 2) == length(age_bands) || error("source matrix columns must match the number of source age bands")

    source = (
        matrix = matrix,
        age_bands = age_bands,
        metadata = (
            source = :prem,
            country = :solomon_islands,
            setting = :all,
            geographic_setting = :all,
            data_source = :v2020,
            orientation = :recipient_source,
        ),
    )

    if data_dir == CONTACT_DATA_DIR
        SOLI_CONTACT_SOURCE_DATA_CACHE[] = source
    end

    return source
end

function adapt_source_band_contact_matrix(
    source_matrix::AbstractMatrix{<:Real},
    source_age_bands::AbstractVector{<:ContactAgeBand};
    max_age::Int = 95,
)
    _validate_contact_age_bands(source_age_bands)
    size(source_matrix, 1) == length(source_age_bands) || error("source_matrix rows must match source_age_bands")
    size(source_matrix, 2) == length(source_age_bands) || error("source_matrix columns must match source_age_bands")
    max_age ≥ 0 || error("max_age must be non-negative")

    target_n = max_age + 1
    target = Matrix{Float64}(undef, target_n, target_n)

    target_indices_by_source = Vector{Vector{Int}}(undef, length(source_age_bands))
    @inbounds for i in eachindex(source_age_bands)
        target_indices_by_source[i] = _source_band_target_indices(source_age_bands[i], max_age)
    end

    @inbounds for recipient_age in 0:max_age
        recipient_source_index = _source_band_index(recipient_age, source_age_bands)
        recipient_row = recipient_age + 1
        for contact_age in 0:max_age
            contact_source_index = _source_band_index(contact_age, source_age_bands)
            contact_col = contact_age + 1
            source_band_targets = target_indices_by_source[contact_source_index]
            target[recipient_row, contact_col] = Float64(source_matrix[recipient_source_index, contact_source_index]) / length(source_band_targets)
        end
    end

    return target
end

function load_solomon_islands_contact_matrix(; data_dir::AbstractString = CONTACT_DATA_DIR, max_age::Int = 95)
    if data_dir == CONTACT_DATA_DIR && max_age == 95
        cached = SOLI_CONTACT_FULL_CACHE[]
        if cached !== nothing
            return cached
        end
    end

    source = load_solomon_islands_contact_source(; data_dir = data_dir)
    contact = adapt_source_band_contact_matrix(source.matrix, source.age_bands; max_age = max_age)

    if data_dir == CONTACT_DATA_DIR && max_age == 95
        SOLI_CONTACT_FULL_CACHE[] = contact
    end

    return contact
end
