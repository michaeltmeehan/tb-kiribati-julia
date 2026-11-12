
"""
    model_outputs(model::DoddTBModel, solution; years, age_breaks=nothing)

Compute annual expected TB outcomes. Results contain `years`, `age_breaks`,
and n_age_groups × n_years matrices. Counts, not rates, are returned.

Incidence, notifications, and TB mortality are inflated from explicitly
modelled pulmonary TB to all-form TB using `model.pulmonary_fraction`.
Infection counters remain unscaled because these describe transmission
within the pulmonary-TB model rather than all-form disease events.
"""
function model_outputs(model::DoddTBModel, solution; years, age_breaks=nothing)
    yy = collect(years)
    layout = state_layout(model)
    extract(c) = annual_events(solution, layout, c; years=yy)

    primary = extract(CumPrimaryInfections)
    reinfections = extract(CumReinfections)
    fast = extract(CumFastProgression)
    slow = extract(CumSlowProgression)
    relapse = extract(CumRelapse)
    treatment = extract(CumTreatmentInitiation)
    untreated_deaths = extract(CumUntreatedTBDeaths)
    treatment_deaths = extract(CumTreatmentTBDeaths)

    η = model.pulmonary_fraction
    incidence = (fast .+ slow .+ relapse) ./ η
    notifications = treatment ./ η
    mortality = (untreated_deaths .+ treatment_deaths) ./ η

    # Aggregate after differencing the single-year age-specific counters.
    group(x) = isnothing(age_breaks) ? x : aggregate_age_groups(x, age_breaks)

    return (
        years = yy,
        age_breaks = isnothing(age_breaks) ? collect(0:layout.nages) : collect(age_breaks),
        incidence = group(incidence),
        notifications = group(notifications),
        mortality = group(mortality),
        primary_infections = group(primary),
        reinfections = group(reinfections),
        fast_progression = group(fast),
        slow_progression = group(slow),
        relapse = group(relapse),
    )
end
