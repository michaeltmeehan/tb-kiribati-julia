
default_output_variables(::DoddExtendedTBModel) = (
    :incidence,
    :notifications,
    :mortality,
    :primary_infections,
    :reinfections,
    :fast_progression,
    :slow_progression,
    :relapse,
    :treatment_completion,
    :self_cure,
    :untreated_mortality,
    :treatment_mortality,
    :new_notifications,
    :recurrent_notifications,
    :pulmonary_notifications,
    :extrapulmonary_notifications,
    :smear_positive_notifications,
    :smear_negative_notifications,
    :new_smear_positive,
    :new_smear_negative,
    :new_extrapulmonary,
    :recurrent_smear_positive,
    :recurrent_smear_negative,
    :recurrent_extrapulmonary,
)

# Directly observed event counters.
const DODD_EXTENDED_COUNTERS = (
    primary_infections=DE.CumPrimaryInfections,
    reinfections=DE.CumReinfections,
    fast_progression=DE.CumFastProgression,
    slow_progression=DE.CumSlowProgression,
    relapse=DE.CumRelapse,
    treatment_completion=DE.CumTreatmentCompletion,
    self_cure=DE.CumSelfCure,
    untreated_mortality=DE.CumUntreatedTBDeaths,
    treatment_mortality=DE.CumTreatmentTBDeaths,
    new_smear_positive=DE.CumNewSP,
    new_smear_negative=DE.CumNewSN,
    new_extrapulmonary=DE.CumNewEP,
    recurrent_smear_positive=DE.CumRecurrentSP,
    recurrent_smear_negative=DE.CumRecurrentSN,
    recurrent_extrapulmonary=DE.CumRecurrentEP,
)

function model_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{V},
) where {V}
    if haskey(DODD_EXTENDED_COUNTERS, V)
        return counter_output(
            ex,
            getproperty(DODD_EXTENDED_COUNTERS, V),
        )
    end

    return extended_combined_output(
        model, ex, Val(V)
    )
end


function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:incidence},
)
    return counter_output(ex, DE.CumFastProgression) .+
        counter_output(ex, DE.CumSlowProgression) .+
        counter_output(ex, DE.CumRelapse)
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:notifications},
)
    return counter_output(
        ex, DE.CumTreatmentInitiation
    )
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:mortality},
)
    return counter_output(ex, DE.CumUntreatedTBDeaths) .+
        counter_output(ex, DE.CumTreatmentTBDeaths)
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:new_notifications},
)
    return counter_output(ex, DE.CumNewSP) .+
        counter_output(ex, DE.CumNewSN) .+
        counter_output(ex, DE.CumNewEP)
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:recurrent_notifications},
)
    return counter_output(ex, DE.CumRecurrentSP) .+
        counter_output(ex, DE.CumRecurrentSN) .+
        counter_output(ex, DE.CumRecurrentEP)
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:pulmonary_notifications},
)
    return counter_output(ex, DE.CumNewSP) .+
        counter_output(ex, DE.CumNewSN) .+
        counter_output(ex, DE.CumRecurrentSP) .+
        counter_output(ex, DE.CumRecurrentSN)
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:extrapulmonary_notifications},
)
    return counter_output(ex, DE.CumNewEP) .+
        counter_output(ex, DE.CumRecurrentEP)
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:smear_positive_notifications},
)
    return counter_output(ex, DE.CumNewSP) .+
        counter_output(ex, DE.CumRecurrentSP)
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:smear_negative_notifications},
)
    return counter_output(ex, DE.CumNewSN) .+
        counter_output(ex, DE.CumRecurrentSN)
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:active_prevalence},
)
    return extended_combined_output(
        model, ex, Val(:untreated_active_prevalence)
    )
end

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{V},
) where {V}
    throw(ArgumentError(
        "Unsupported DoddExtended output: $V"
    ))
end


# ------------------------------------------------------------
# Direct epidemiological events
# ------------------------------------------------------------

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:disease_onsets},
)
    return counter_output(ex, DE.CumFastProgression) .+
        counter_output(ex, DE.CumSlowProgression) .+
        counter_output(ex, DE.CumRelapse)
end


function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_initiations},
)
    return counter_output(ex, DE.CumTreatmentInitiation)
end


function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:tb_deaths},
)
    return counter_output(ex, DE.CumUntreatedTBDeaths) .+
        counter_output(ex, DE.CumTreatmentTBDeaths)
end


function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_deaths},
)
    return counter_output(ex, DE.CumTreatmentTBDeaths)
end


# ------------------------------------------------------------
# Epidemiological stocks
# ------------------------------------------------------------

function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:untreated_active_prevalence},
)
    return stock_output(ex, DE.I0SP:DE.I1EP)
end


function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:latent_prevalence},
)
    return stock_output(
        ex,
        (DE.E0, DE.E1, DE.L0, DE.L1),
    )
end


function extended_combined_output(
    model::DoddExtendedTBModel,
    ex::OutputExtraction,
    ::Val{:treatment_prevalence},
)
    return stock_output(ex, (DE.T,))
end
