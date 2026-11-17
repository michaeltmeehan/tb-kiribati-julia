
"""
    model_outputs(
        model::DoddExtendedTBModel,
        solution;
        years,
        age_breaks=nothing,
    )

Return annual outcomes as age-group × year matrices.

All TB presentations are explicitly represented.
No pulmonary-fraction correction is applied.
"""
function model_outputs(
    model::DoddExtendedTBModel,
    solution;
    years,
    age_breaks=nothing,
)
    yy = collect(years)
    layout = state_layout(model)

    extract(c) = annual_events(
        solution, layout, c; years=yy
    )

    primary = extract(DE.CumPrimaryInfections)
    reinfections = extract(DE.CumReinfections)

    fast = extract(DE.CumFastProgression)
    slow = extract(DE.CumSlowProgression)
    relapse = extract(DE.CumRelapse)

    notifications =
        extract(DE.CumTreatmentInitiation)

    completion =
        extract(DE.CumTreatmentCompletion)

    untreated_deaths =
        extract(DE.CumUntreatedTBDeaths)

    treatment_deaths =
        extract(DE.CumTreatmentTBDeaths)

    self_cure =
        extract(DE.CumSelfCure)

    new_sp = extract(DE.CumNewSP)
    new_sn = extract(DE.CumNewSN)
    new_ep = extract(DE.CumNewEP)

    rec_sp = extract(DE.CumRecurrentSP)
    rec_sn = extract(DE.CumRecurrentSN)
    rec_ep = extract(DE.CumRecurrentEP)

    incidence = fast .+ slow .+ relapse
    mortality = untreated_deaths .+ treatment_deaths

    new_notifications =
        new_sp .+ new_sn .+ new_ep

    recurrent_notifications =
        rec_sp .+ rec_sn .+ rec_ep

    smear_positive = new_sp .+ rec_sp
    smear_negative = new_sn .+ rec_sn
    extrapulmonary = new_ep .+ rec_ep

    pulmonary = smear_positive .+ smear_negative

    group(x) = isnothing(age_breaks) ?
               x :
               aggregate_age_groups(x, age_breaks)

    return (
        years=yy,
        age_breaks=isnothing(age_breaks) ?
                   collect(0:layout.nages) :
                   collect(age_breaks),
        incidence=group(incidence),
        notifications=group(notifications),
        mortality=group(mortality),
        primary_infections=group(primary),
        reinfections=group(reinfections),
        fast_progression=group(fast),
        slow_progression=group(slow),
        relapse=group(relapse),
        treatment_completion=group(completion),
        self_cure=group(self_cure),
        untreated_mortality=group(untreated_deaths),
        treatment_mortality=group(treatment_deaths),
        new_notifications=group(new_notifications),
        recurrent_notifications=
        group(recurrent_notifications),
        pulmonary_notifications=group(pulmonary),
        extrapulmonary_notifications=
        group(extrapulmonary),
        smear_positive_notifications=
        group(smear_positive),
        smear_negative_notifications=
        group(smear_negative),
        new_smear_positive=group(new_sp),
        new_smear_negative=group(new_sn),
        new_extrapulmonary=group(new_ep),
        recurrent_smear_positive=group(rec_sp),
        recurrent_smear_negative=group(rec_sn),
        recurrent_extrapulmonary=group(rec_ep),
    )
end
