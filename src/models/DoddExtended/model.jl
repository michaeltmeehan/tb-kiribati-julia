
module DoddExtendedStates

# Epidemiological compartments
const S = 1
const E0 = 2
const E1 = 3
const L0 = 4
const L1 = 5
const I0SP = 6
const I0SN = 7
const I0EP = 8
const I1SP = 9
const I1SN = 10
const I1EP = 11
const T = 12
const R = 13

# Cumulative event counters
const CumPrimaryInfections = 14
const CumReinfections = 15
const CumFastProgression = 16
const CumSlowProgression = 17
const CumRelapse = 18
const CumTreatmentInitiation = 19
const CumTreatmentCompletion = 20
const CumUntreatedTBDeaths = 21
const CumTreatmentTBDeaths = 22
const CumSelfCure = 23
const CumNewSP = 24
const CumNewSN = 25
const CumNewEP = 26
const CumRecurrentSP = 27
const CumRecurrentSN = 28
const CumRecurrentEP = 29

end

const DE = DoddExtendedStates


struct DoddExtendedTBModel <: AbstractTBModel
    layout::StateLayout
    contact::Matrix{Float64}
    infectiousness::Vector{Float64}
    presentation::Matrix{Float64}
    reinfection::Bool
end


function DoddExtendedTBModel(
    contact::AbstractMatrix,
    presentation::AbstractMatrix;
    reinfection::Bool=true,
)
    # -------------------------------------------------------
    # Contact matrix and infectiousness
    # -------------------------------------------------------

    contact_matrix = normalise_model_contacts(contact)
    nages = size(contact_matrix, 1)

    infectiousness = adult_infectiousness(nages)

    # -------------------------------------------------------
    # Disease presentation probabilities
    # -------------------------------------------------------

    size(presentation) == (nages, 3) ||
        throw(DimensionMismatch(
            "Presentation must have size ($nages, 3)"
        ))

    P = Matrix{Float64}(presentation)

    all(x -> isfinite(x) && x >= 0, P) ||
        throw(ArgumentError(
            "Presentation probabilities must be finite and nonnegative"
        ))

    all(
        a -> isapprox(
            sum(@view P[a, :]),
            1.0;
            atol=1e-10,
        ),
        1:nages,
    ) || throw(ArgumentError(
        "Presentation probabilities must sum to one for every age"
    ))

    # -------------------------------------------------------
    # Model construction
    # -------------------------------------------------------

    return DoddExtendedTBModel(
        StateLayout(13, 16, nages),
        contact_matrix,
        infectiousness,
        P,
        reinfection,
    )
end


state_layout(model::DoddExtendedTBModel) = model.layout
