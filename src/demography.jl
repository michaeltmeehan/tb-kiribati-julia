const DEMOGRAPHIC_AGEING_RATE = 1.0

struct DemographicSchedule
    years::Vector{Int}
    mortality::Matrix{Float64}
    fertility::Matrix{Float64}
    migration::Matrix{Float64}
    births::Union{Nothing,Vector{Float64}}
    fertility_mode::Symbol
    first_year::Int
    last_year::Int
end

function _validate_demographic_matrix(name::AbstractString, matrix::AbstractMatrix{<:Real}, nyears::Int)
    size(matrix, 1) == NAGE || error("$name must have 96 age rows")
    size(matrix, 2) == nyears || error("$name must have one column per simulation year")
    all(isfinite, matrix) || error("$name must contain only finite values")
    return nothing
end

function DemographicSchedule(
    years::AbstractVector{<:Integer},
    mortality::AbstractMatrix{<:Real},
    fertility::AbstractMatrix{<:Real},
    migration::AbstractMatrix{<:Real};
    births::Union{Nothing,AbstractVector{<:Real}} = nothing,
    fertility_mode::Symbol = :agepi_compatible,
)
    nyears = length(years)
    nyears > 0 || error("years must not be empty")
    fertility_mode in (:agepi_compatible, :wpp) || error("unsupported fertility_mode: $fertility_mode")

    yrs = Int[years...]
    for i in 2:nyears
        yrs[i] > yrs[i - 1] || error("years must be strictly increasing")
        yrs[i] == yrs[i - 1] + 1 || error("years must be consecutive annual values")
    end

    mort = Matrix{Float64}(mortality)
    fert = Matrix{Float64}(fertility)
    mig = Matrix{Float64}(migration)

    _validate_demographic_matrix("mortality", mort, nyears)
    _validate_demographic_matrix("fertility", fert, nyears)
    _validate_demographic_matrix("migration", mig, nyears)

    any(<(0.0), mort) && error("mortality must be non-negative")
    any(<(0.0), fert) && error("fertility must be non-negative")

    birth_vec = if births === nothing
        nothing
    else
        vec = Float64[births...]
        length(vec) == nyears || error("births must have one value per simulation year")
        any(<(0.0), vec) && error("births must be non-negative")
        vec
    end

    fertility_mode === :wpp && birth_vec === nothing && error("births must be provided when fertility_mode = :wpp")

    return DemographicSchedule(yrs, mort, fert, mig, birth_vec, fertility_mode, yrs[1], yrs[end])
end

@inline function demographic_births(schedule::DemographicSchedule, year_idx::Int, age_pop::AbstractVector)
    if schedule.fertility_mode === :wpp
        births = schedule.births
        births === nothing && error("wpp fertility mode requires births to be populated")
        return births[year_idx]
    elseif schedule.fertility_mode === :agepi_compatible
        total = 0.0
        @inbounds for a in 1:NAGE
            total += schedule.fertility[a, year_idx] * age_pop[a]
        end
        return total
    else
        error("unsupported fertility_mode: $(schedule.fertility_mode)")
    end
end

@inline function demographic_year_index(schedule::DemographicSchedule, t::Real)
    year = floor(Int, t)
    return clamp(year - schedule.first_year + 1, 1, length(schedule.years))
end

@inline function demographic_tstops(schedule::DemographicSchedule)
    return Float64.(schedule.years[2:end])
end

@inline function age_population!(age_pop::AbstractVector{<:Real}, u::AbstractVector)
    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        total = 0.0
        for j in 1:NEPI
            total += u[base + j]
        end
        age_pop[a] = total
    end
    return age_pop
end

@inline function apply_proportional_migration!(du, u, age_base::Int, na::Real, migration::Real)
    if migration == 0.0
        return nothing
    end

    if na > 0.0
        scale = migration / na
        @inbounds for j in 1:NEPI
            du[age_base + j] += scale * u[age_base + j]
        end
    elseif migration > 0.0
        @inbounds du[age_base + MtbNaive] += migration
    end
    return nothing
end

function apply_demography!(du, u, p, t)
    schedule = p.demography
    schedule === nothing && return nothing

    year_idx = demographic_year_index(schedule, t)
    age_pop = p.tmp_age_pop
    age_population!(age_pop, u)

    du[MtbNaive] += demographic_births(schedule, year_idx, age_pop)

    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        na = age_pop[a]
        mu = schedule.mortality[a, year_idx]
        mig = schedule.migration[a, year_idx]

        if mu != 0.0
            for j in 1:NEPI
                du[base + j] -= mu * u[base + j]
            end
        end

        apply_proportional_migration!(du, u, base, na, mig)
    end

    if p.ageing_enabled
        @inbounds for a in 1:(NAGE - 1)
            base = (a - 1) * NSTATE
            next = base + NSTATE
            for j in 1:NEPI
                flow = DEMOGRAPHIC_AGEING_RATE * u[base + j]
                du[base + j] -= flow
                du[next + j] += flow
            end
        end
    end

    return nothing
end

function synthetic_demographic_schedule()
    years = collect(2025:2030)
    nyears = length(years)
    mortality = zeros(Float64, NAGE, nyears)
    fertility = zeros(Float64, NAGE, nyears)
    migration = zeros(Float64, NAGE, nyears)

    @inbounds for yi in 1:nyears
        year = years[yi]
        year_offset = year - years[1]
        mortality_scale = 1.0 + 0.02 * year_offset
        fertility_scale = 1.0 - 0.01 * year_offset

        for age in 0:(NAGE - 1)
            idx = age + 1
            mortality[idx, yi] = mortality_scale * begin
                if age <= 49
                    0.005
                elseif age <= 64
                    0.010
                elseif age <= 79
                    0.030
                else
                    0.100
                end
            end

            fertility[idx, yi] = if 15 <= age <= 49
                0.005 * fertility_scale
            else
                0.0
            end
        end

        migration[11, yi] = 5.0 + year_offset
        migration[36, yi] = -2.0 - 0.25 * year_offset
        migration[61, yi] = 1.0 + 0.5 * year_offset
        migration[81, yi] = -0.5
        migration[91, yi] = 0.75
    end

    return DemographicSchedule(years, mortality, fertility, migration)
end
