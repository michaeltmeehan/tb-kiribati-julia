
const RG = RagonnetIndices

default_output_variables(::RagonnetTBModel) = (
    :infection_events,
    :disease_onsets,
    :treatment_initiations,
    :treatment_completions,
    :treatment_relapses,
    :tb_deaths,
)

# ------------------------------------------------------------
# Direct event counters
# ------------------------------------------------------------

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:infection_events},
)
    return counter_output(ex, RG.CumInfectionsOther) .+
        counter_output(ex, RG.CumInfectionsContained)
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:progression_to_active_tb},
)
    return counter_output(
        ex, RG.CumProgressionToActiveTB
    )
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_relapses},
)
    return counter_output(ex, RG.CumRelapseTB)
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:disease_onsets},
)
    return counter_output(
        ex, RG.CumProgressionToActiveTB
    ) .+ counter_output(ex, RG.CumRelapseTB)
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_initiations},
)
    return counter_output(
        ex, RG.CumTreatmentInitiation
    )
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_completions},
)
    return counter_output(
        ex, RG.CumTreatmentCompletion
    )
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:tb_deaths},
)
    return counter_output(ex, RG.CumDeathsTB)
end

# ------------------------------------------------------------
# Compatibility output names
# ------------------------------------------------------------

function model_output(
    model::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:incidence},
)
    return model_output(model, ex, Val(:disease_onsets))
end

function model_output(
    model::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:notifications},
)
    return model_output(
        model, ex, Val(:treatment_initiations)
    )
end

function model_output(
    model::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:mortality},
)
    return model_output(model, ex, Val(:tb_deaths))
end

# ------------------------------------------------------------
# Epidemiological stocks
# ------------------------------------------------------------

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:untreated_active_prevalence},
)
    return stock_output(
        ex,
        (
            RG.SubClinLow,
            RG.SubClinInf,
            RG.ClinLow,
            RG.ClinInf,
        ),
    )
end

function model_output(
    model::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:active_prevalence},
)
    return model_output(
        model, ex, Val(:untreated_active_prevalence)
    )
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_prevalence},
)
    return stock_output(ex, (RG.Treatment,))
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:contained_prevalence},
)
    return stock_output(ex, (RG.Contained,))
end

function model_output(
    ::RagonnetTBModel,
    ex::OutputExtraction,
    ::Val{:incipient_prevalence},
)
    return stock_output(ex, (RG.Incipient,))
end
