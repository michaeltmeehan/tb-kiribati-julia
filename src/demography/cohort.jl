
using CSV

"""
    CohortDemography(data; nages=96, ...)

Independent cohort-component demographic projection.

Annual population changes arise from:
- age-specific fertility;
- background mortality;
- cohort ageing;
- net migration.

No reconciliation to future WPP populations is performed.

WPP population data are used only for initialisation.
"""
struct CohortDemography <: AbstractDemography
    data::DemographicData
    nages::Int
    female_fraction::Float64
    migration_scheme::Symbol
    age_migration::Union{Nothing,Matrix{Float64}}
    migration_first_year::Int
end


function CohortDemography(
    data::DemographicData;
    nages::Int=96,
    female_fraction::Real=0.5,
    migration_scheme::Symbol=:proportional,
    age_migration::Union{Nothing,AbstractMatrix}=nothing,
    migration_first_year::Int=2024,
)
    2 <= nages <= size(data.population, 1) ||
        throw(ArgumentError("Invalid number of age classes"))

    0 <= female_fraction <= 1 ||
        throw(ArgumentError(
            "female_fraction must lie in [0, 1]"
        ))

    migration_scheme in (:proportional, :age) ||
        throw(ArgumentError(
            "migration_scheme must be :proportional or :age"
        ))

    if migration_scheme == :age
        isnothing(age_migration) &&
            throw(ArgumentError(
                "Age-specific migration data are required"
            ))

        size(age_migration, 1) ==
            size(data.population, 1) ||
            throw(DimensionMismatch(
                "Age migration must contain all source ages"
            ))

        size(age_migration, 2) > 0 ||
            throw(ArgumentError(
                "Age migration must contain at least one year"
            ))

        all(isfinite, age_migration) ||
            throw(ArgumentError(
                "Nonfinite age-specific migration"
            ))
    end

    return CohortDemography(
        data,
        nages,
        Float64(female_fraction),
        migration_scheme,
        isnothing(age_migration) ?
        nothing : Matrix{Float64}(age_migration),
        migration_first_year,
    )
end


"""
    load_age_migration(path; first_year=2024, last_year=2100)

Read the WPP `migration_age.csv` extract.

Required columns: year, age, mig

The source `mig` values are in thousands; the returned
matrix contains individuals.

Rows correspond to ages 0, 1, ..., 100.
Columns correspond to successive calendar years.
"""
function load_age_migration(
    path::AbstractString;
    first_year::Int=2024,
    last_year::Int=2100,
)
    first_year <= last_year ||
        throw(ArgumentError("Invalid year range"))

    table = CSV.File(path)
    required = (:year, :age, :mig)

    all(k -> k in propertynames(table), required) ||
        throw(ArgumentError(
            "Migration file requires year, age, mig"
        ))

    result = fill(
        NaN, 101, last_year - first_year + 1
    )

    for row in table
        y = Int(row.year)
        first_year <= y <= last_year || continue

        a = Int(row.age)
        0 <= a <= 100 ||
            throw(ArgumentError("Invalid migration age"))

        idx = (a + 1, y - first_year + 1)

        isnan(result[idx...]) ||
            throw(ArgumentError(
                "Duplicate migration year/age"
            ))

        result[idx...] = 1000 * Float64(row.mig)
    end

    all(isfinite, result) ||
        throw(ArgumentError(
            "Missing or nonfinite migration observations"
        ))

    return result
end


"""
WPP initial population at the requested year.

This method does not return an independently projected
future population. Future states are obtained by simulation.
"""
function get_population(
    demography::CohortDemography,
    year::Real,
)
    return get_population(
        demography.data,
        year;
        nages=demography.nages,
    )
end


"""
Return age-specific net migration for the transition
from `year` to `year + 1`.

Net migration is returned in individuals.
"""
function cohort_migration(
    demography::CohortDemography,
    year::Integer,
    starting_population::AbstractVector,
    aged_population::AbstractVector,
)
    nages = demography.nages

    length(starting_population) == nages ||
        throw(DimensionMismatch("Invalid starting population"))

    length(aged_population) == nages ||
        throw(DimensionMismatch("Invalid aged population"))

    destination_year = year + 1

    national = migration_flow(
        demography.data, destination_year
    )

    T = promote_type(
        eltype(starting_population),
        eltype(aged_population),
        Float64,
    )

    M = zeros(T, nages)

    if demography.migration_scheme == :proportional
        total = sum(aged_population)
        total > 0 ||
            throw(ArgumentError("Zero population"))

        M .= national .* aged_population ./ total

    else
        j = destination_year -
            demography.migration_first_year + 1

        1 <= j <= size(demography.age_migration, 2) ||
            throw(ArgumentError(
                "Age-specific migration unavailable for $destination_year"
            ))

        raw = @view demography.age_migration[:, j]

        M[1:(end-1)] .= raw[1:(nages-1)]
        M[end] = sum(@view raw[nages:end])

        # Match national migration totals while retaining
        # the age-specific profile.
        total = sum(starting_population)
        total > 0 ||
            throw(ArgumentError("Zero starting population"))

        M .+= (national - sum(M)) .*
            starting_population ./ total
    end

    return M
end


"""
    apply_demography!(u, model, demography::CohortDemography, year)

Independent annual demographic update from `year` to
`year + 1`, with no WPP population reconciliation.

Disease transitions are handled by the ODE between
demographic updates.

Cumulative counters are not modified.
"""
function apply_demography!(
    u::AbstractVector,
    model::AbstractTBModel,
    demography::CohortDemography,
    year::Real;
    susceptible_compartment::Int=1,
)
    isinteger(year) ||
        throw(ArgumentError(
            "Cohort updates require integer calendar years"
        ))

    y = Int(year)
    data = demography.data
    destination_year = y + 1

    data.first_year <= y &&
        destination_year <= data.last_year ||
        throw(ArgumentError(
            "Demographic data unavailable for this transition"
        ))

    layout = state_layout(model)
    nages = layout.nages
    nepi = layout.nepi
    stride = nstates(layout)

    nages == demography.nages ||
        throw(DimensionMismatch(
            "Model and demography age dimensions differ"
        ))

    length(u) == nages * stride ||
        throw(DimensionMismatch(
            "Invalid state vector length"
        ))

    1 <= susceptible_compartment <= nepi ||
        throw(ArgumentError(
            "Invalid susceptible compartment"
        ))

    all(isfinite, u) ||
        throw(ArgumentError("Nonfinite state vector"))

    T = promote_type(eltype(u), Float64)

    # --------------------------------------------------
    # 1. Record starting populations
    # --------------------------------------------------

    starting = zeros(T, nages)
    updated = zeros(T, nages, nepi)

    for a in 1:nages
        b = (a - 1) * stride

        for c in 1:nepi
            value = u[b+c]

            value >= 0 ||
                throw(ArgumentError(
                    "Negative living population"
                ))

            starting[a] += value
        end
    end

    # --------------------------------------------------
    # 2. Mortality and cohort ageing
    # --------------------------------------------------

    for a in 1:nages
        survival = exp(
            -mortality_rate(data, a-1, destination_year)
        )

        destination = min(a+1, nages)
        b = (a-1) * stride

        for c in 1:nepi
            updated[destination, c] +=
                survival * u[b+c]
        end
    end

    aged = vec(sum(updated; dims=2))

    # --------------------------------------------------
    # 3. Determine net migration by age
    # --------------------------------------------------

    M = cohort_migration(
        demography, y, starting, aged
    )

    # Population after migration, before newborns.
    # This determines midyear fertility exposure.
    premirth = aged .+ M

    # --------------------------------------------------
    # 4. Births from midyear fertility exposure
    # --------------------------------------------------

    births = zero(T)

    for a in 1:nages
        age = a-1

        row = age - data.first_fertility_age + 1

        if 1 <= row <= size(data.fertility, 1)
            rate = data.fertility[
                row,
                demographic_year_index(
                    data, destination_year
                ),
            ]

            exposure = (starting[a] + premirth[a]) / 2

            exposure >= 0 ||
                throw(ArgumentError(
                    "Negative fertility exposure"
                ))

            births += demography.female_fraction *
                rate * exposure
        end
    end

    infant_survival = exp(
        -mortality_rate(data, 0, destination_year) / 2
    )

    updated[1, susceptible_compartment] +=
        births * infant_survival

    # --------------------------------------------------
    # 5. Apply migration to disease compartments
    # --------------------------------------------------

    for a in 1:nages
        migration = M[a]

        if migration >= 0
            # Incoming migrants enter susceptible.
            updated[a, susceptible_compartment] +=
                migration
        else
            # Emigrants are removed proportionally
            # from all living compartments.
            current = sum(@view updated[a, :])

            current >= -migration ||
                throw(ArgumentError(
                    "Emigration exceeds population at age $(a-1)"
                ))

            current > 0 ||
                throw(ArgumentError(
                    "Cannot remove migrants from empty age class"
                ))

            factor = (current + migration) / current

            for c in 1:nepi
                updated[a, c] *= factor
            end
        end
    end

    # --------------------------------------------------
    # 6. Write living states back
    # --------------------------------------------------

    for a in 1:nages
        b = (a-1) * stride

        for c in 1:nepi
            u[b+c] = updated[a, c]
        end
    end

    # Cumulative counters remain unchanged.
    return nothing
end
