# src/demography/operators.jl

"""
    target_population(demography, year)

Return the target age-specific population after the annual
demographic update from `year` to `year + 1`.
"""
function target_population end


"""
    age_and_survive!(updated, u, layout, data, year)

Apply background mortality and whole-cohort ageing.

Every surviving cohort advances one age class. The final
age class is open-ended.

Only epidemiological compartments are processed.
"""
function age_and_survive!(
    updated::AbstractMatrix,
    u::AbstractVector,
    layout::StateLayout,
    data::DemographicData,
    year::Real,
)
    nages = layout.nages
    nepi = layout.nepi
    nstate = nstates(layout)

    size(updated) == (nages, nepi) ||
        throw(DimensionMismatch("Incorrect workspace dimensions"))

    length(u) == nages * nstate ||
        throw(DimensionMismatch("Incorrect state vector length"))

    fill!(updated, 0.0)

    @inbounds for a in 1:nages
        survival = exp(-mortality_rate(data, a - 1, year))
        destination = min(a + 1, nages)
        base = (a - 1) * nstate

        for c in 1:nepi
            updated[destination, c] +=
                survival * u[base + c]
        end
    end

    return nothing
end


"""
    apply_demography!(u, model, demography, year;
                      susceptible_compartment=1)

Apply an annual demographic update:

1. Background mortality.
2. Whole-cohort ageing.
3. Reconciliation to the scenario-specific target population.

The update advances the population from `year` to `year + 1`.

Only epidemiological compartments are modified.
Cumulative event counters remain unchanged.
"""
function apply_demography!(
    u::AbstractVector,
    model::AbstractTBModel,
    demography::AbstractDemography,
    year::Real;
    susceptible_compartment::Int = 1,
)
    layout = state_layout(model)

    nages = layout.nages
    nepi = layout.nepi
    nstate = nstates(layout)

    # --------------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------------

    length(u) == nages * nstate ||
        throw(DimensionMismatch(
            "State vector length does not match model layout"
        ))

    1 <= susceptible_compartment <= nepi ||
        throw(ArgumentError(
            "Invalid susceptible compartment"
        ))

    all(isfinite, u) ||
        throw(ArgumentError("Nonfinite state vector"))

    @inbounds for a in 1:nages
        base = (a - 1) * nstate

        for c in 1:nepi
            u[base + c] >= 0 ||
                throw(ArgumentError(
                    "Negative epidemiological population"
                ))
        end
    end

    # --------------------------------------------------------
    # 2. Scenario-specific target population
    # --------------------------------------------------------

    target = target_population(demography, year)

    length(target) == nages ||
        throw(DimensionMismatch(
            "Target population does not match model age classes"
        ))

    all(x -> isfinite(x) && x >= 0, target) ||
        throw(ArgumentError("Invalid target population"))

    # --------------------------------------------------------
    # 3. Background mortality and whole-cohort ageing
    # --------------------------------------------------------

    updated = zeros(Float64, nages, nepi)

    age_and_survive!(
        updated,
        u,
        layout,
        demography.data,
        year,
    )

    # --------------------------------------------------------
    # 4. Reconcile to target population
    # --------------------------------------------------------

    @inbounds for a in 1:nages
        current = 0.0

        for c in 1:nepi
            current += updated[a, c]
        end

        desired = target[a]

        if current > 0
            scale = desired / current

            for c in 1:nepi
                updated[a, c] *= scale
            end
        else
            for c in 1:nepi
                updated[a, c] = 0.0
            end

            updated[a, susceptible_compartment] = desired
        end
    end

    # --------------------------------------------------------
    # 5. Write epidemiological compartments back
    # --------------------------------------------------------

    @inbounds for a in 1:nages
        base = (a - 1) * nstate

        for c in 1:nepi
            u[base + c] = updated[a, c]
        end
    end

    return nothing
end