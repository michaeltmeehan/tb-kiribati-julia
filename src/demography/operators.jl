
"""
    target_population(demography, year)

Return the age-specific population target after advancing
the demographic scenario from `year` to `year + 1`.
"""
function target_population end


"""
    age_and_survive!(updated, u, layout, data, year)

Apply background mortality and whole-cohort ageing.

The final age class is open-ended. Age zero is initially
empty and is subsequently populated during reconciliation.

Only living epidemiological compartments are processed.
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
    stride = nstates(layout)

    size(updated) == (nages, nepi) ||
        throw(DimensionMismatch(
            "Incorrect demographic workspace dimensions"
        ))

    length(u) == nages * stride ||
        throw(DimensionMismatch(
            "State vector length does not match layout"
        ))

    fill!(updated, zero(eltype(updated)))

    @inbounds for a in 1:nages
        survival = exp(
            -mortality_rate(data, a - 1, year)
        )

        destination = min(a + 1, nages)
        base = (a - 1) * stride

        for c in 1:nepi
            updated[destination, c] +=
                survival * u[base+c]
        end
    end

    return updated
end


"""
    reconcile_population!(updated, target;
                          susceptible_compartment=1)

Reconcile living populations to age-specific targets.

For each age:
- If the population is positive, scale every compartment
  by the same factor.
- If the population is zero, put the entire target
  population in the susceptible compartment.

This preserves within-age compartment proportions
whenever the pre-reconciliation population is positive.
"""
function reconcile_population!(
    updated::AbstractMatrix,
    target::AbstractVector;
    susceptible_compartment::Int=1,
)
    nages, nepi = size(updated)

    length(target) == nages ||
        throw(DimensionMismatch(
            "Target population has incorrect length"
        ))

    1 <= susceptible_compartment <= nepi ||
        throw(ArgumentError(
            "Invalid susceptible compartment"
        ))

    all(x -> isfinite(x) && x >= 0, updated) ||
        throw(ArgumentError(
            "Invalid epidemiological population"
        ))

    all(x -> isfinite(x) && x >= 0, target) ||
        throw(ArgumentError(
            "Invalid target population"
        ))

    T = eltype(updated)

    @inbounds for a in 1:nages
        current = zero(T)

        for c in 1:nepi
            current += updated[a, c]
        end

        desired = target[a]

        if current > zero(T)
            scale = desired / current

            for c in 1:nepi
                updated[a, c] *= scale
            end
        else
            for c in 1:nepi
                updated[a, c] = zero(T)
            end

            updated[a, susceptible_compartment] = desired
        end
    end

    return updated
end


"""
    apply_demography!(u, model, demography, year;
                      susceptible_compartment=1)

Apply the annual demographic update from `year`
to `year + 1`.

1. Background mortality.
2. Whole-cohort ageing.
3. Reconciliation to the scenario population.

Cumulative event counters remain unchanged.

Births and migration are not explicitly simulated;
their net demographic effects are represented through
population reconciliation.
"""
function apply_demography!(
    u::AbstractVector,
    model::AbstractTBModel,
    demography::AbstractDemography,
    year::Real;
    susceptible_compartment::Int=1,
)
    layout = state_layout(model)

    nages = layout.nages
    nepi = layout.nepi
    stride = nstates(layout)

    length(u) == nages * stride ||
        throw(DimensionMismatch(
            "State vector length does not match model"
        ))

    1 <= susceptible_compartment <= nepi ||
        throw(ArgumentError(
            "Invalid susceptible compartment"
        ))

    all(isfinite, u) ||
        throw(ArgumentError(
            "State vector contains nonfinite values"
        ))

    @inbounds for a in 1:nages
        base = (a - 1) * stride

        for c in 1:nepi
            u[base+c] >= 0 ||
                throw(ArgumentError(
                    "Negative epidemiological population"
                ))
        end
    end

    target = target_population(demography, year)

    length(target) == nages ||
        throw(DimensionMismatch(
            "Target population does not match model ages"
        ))

    # Preserve differentiated state values in the
    # demographic workspace.
    T = promote_type(
        eltype(u),
        eltype(target),
        Float64,
    )

    updated = zeros(T, nages, nepi)

    age_and_survive!(
        updated,
        u,
        layout,
        demography.data,
        demographic_mortality_year(demography, year),
    )

    reconcile_population!(
        updated,
        target;
        susceptible_compartment,
    )

    # Write back living compartments only.
    @inbounds for a in 1:nages
        base = (a - 1) * stride

        for c in 1:nepi
            u[base+c] = updated[a, c]
        end
    end

    return nothing
end

"""
Reference year used for background mortality during an
annual demographic update.

Equilibrium demography freezes mortality at its reference
year. Projected demography uses the evolving calendar year.
"""
demographic_mortality_year(
    ::AbstractDemography,
    year::Real,
) = year

demographic_mortality_year(
    demography::EquilibriumDemography,
    year::Real,
) = demography.reference_year
