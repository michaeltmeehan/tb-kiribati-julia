
default_output_variables(::DoddTBModel) = (
    :incidence,
    :notifications,
    :mortality,
    :primary_infections,
    :reinfections,
    :fast_progression,
    :slow_progression,
    :relapse,
)

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:primary_infections},
)
    return counter_output(ex, CumPrimaryInfections)
end

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:reinfections},
)
    return counter_output(ex, CumReinfections)
end

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:fast_progression},
)
    return counter_output(ex, CumFastProgression)
end

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:slow_progression},
)
    return counter_output(ex, CumSlowProgression)
end

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:relapse},
)
    return counter_output(ex, CumRelapse)
end

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:incidence},
)
    fast = counter_output(ex, CumFastProgression)
    slow = counter_output(ex, CumSlowProgression)
    relapse = counter_output(ex, CumRelapse)

    return (fast .+ slow .+ relapse) ./
        model.pulmonary_fraction
end

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:notifications},
)
    return counter_output(
        ex, CumTreatmentInitiation
    ) ./ model.pulmonary_fraction
end

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:mortality},
)
    untreated = counter_output(
        ex, CumUntreatedTBDeaths
    )

    treated = counter_output(
        ex, CumTreatmentTBDeaths
    )

    return (untreated .+ treated) ./
        model.pulmonary_fraction
end

function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:active_prevalence},
)
    return model_output(
        model, ex, Val(:untreated_active_prevalence)
    )
end


# ------------------------------------------------------------
# Direct epidemiological events
# ------------------------------------------------------------

"""
Active TB disease onsets, including relapse.

This is the directly modelled pulmonary disease-onset
count, without all-form extrapolation.
"""
function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:disease_onsets},
)
    return counter_output(ex, CumFastProgression) .+
        counter_output(ex, CumSlowProgression) .+
        counter_output(ex, CumRelapse)
end


"""
Direct treatment initiations, without correction for the
unmodelled extrapulmonary disease fraction.
"""
function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_initiations},
)
    return counter_output(ex, CumTreatmentInitiation)
end


"""
Direct TB-attributable deaths in the explicitly modelled
pulmonary disease population.
"""
function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:tb_deaths},
)
    return counter_output(ex, CumUntreatedTBDeaths) .+
        counter_output(ex, CumTreatmentTBDeaths)
end


function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_deaths},
)
    return counter_output(ex, CumTreatmentTBDeaths)
end


# ------------------------------------------------------------
# Epidemiological stocks
# ------------------------------------------------------------

"""
Untreated active TB population at the post-demographic
state of each requested calendar year.
"""
function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:untreated_active_prevalence},
)
    return stock_output(ex, (I,))
end


"""
Early and late latent infection populations.
"""
function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:latent_prevalence},
)
    return stock_output(ex, (E, L))
end


"""
Population currently receiving TB treatment.
"""
function model_output(
    model::DoddTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_prevalence},
)
    return stock_output(ex, (T,))
end

