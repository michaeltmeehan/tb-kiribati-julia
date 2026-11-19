# ------------------------------------------------------------
# Compartment indices
# ------------------------------------------------------------

const S = 1  # Susceptible
const E = 2  # Early latent
const L = 3  # Late latent
const I = 4  # Active TB
const T = 5  # Treatment
const R = 6  # Recovered

# ------------------------------------------------------------
# Cumulative event indices
# ------------------------------------------------------------

const CumPrimaryInfections = 7
const CumReinfections = 8
const CumFastProgression = 9
const CumSlowProgression = 10
const CumRelapse = 11
const CumTreatmentInitiation = 12
const CumTreatmentCompletion = 13
const CumUntreatedTBDeaths = 14
const CumTreatmentTBDeaths = 15
const CumSelfCure = 16

# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

struct DoddTBModel <: AbstractTBModel
    layout::StateLayout
    contact::Matrix{Float64}
    infectiousness::Vector{Float64}
    pulmonary_fraction::Float64
end


function DoddTBModel(
    contact::Matrix{Float64};
    pulmonary_fraction::Float64=0.83,
)
    nages = size(contact, 1)

    size(contact, 2) == nages ||
        throw(ArgumentError("Contact matrix must be square"))

    nages > 0 ||
        throw(ArgumentError("Contact matrix must not be empty"))

    all(x -> isfinite(x) && x >= 0, contact) ||
        throw(ArgumentError(
            "Contact matrix must contain finite, nonnegative values"
        ))

    0 < pulmonary_fraction <= 1 ||
        throw(ArgumentError(
            "Pulmonary fraction must lie in (0, 1]"
        ))

    nages = size(contact, 1)

    normalised_contact = normalise_model_contacts(contact)
    infectiousness = adult_infectiousness(nages)
    layout = StateLayout(6, 10, nages)

    return DoddTBModel(
        layout,
        normalised_contact,
        infectiousness,
        pulmonary_fraction,
    )
end


state_layout(model::DoddTBModel) = model.layout