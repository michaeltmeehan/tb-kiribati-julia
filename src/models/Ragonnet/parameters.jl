
struct RagonnetTBParameters
    contact::Matrix{Float64}
    beta::Float64
    susceptibility_naive::Vector{Float64}
    susceptibility_contained::Vector{Float64}
    susceptibility_cleared::Vector{Float64}
    susceptibility_recovered::Vector{Float64}
    containment_rate::Vector{Float64}
    progression_rate::Vector{Float64}
    infectiousness_weights::NTuple{4,Float64}
    clearance_rate::Float64
    breakdown_rate::Float64
    prop_infectious::Float64
    clinical_progression_rate::Float64
    clinical_regression_rate::Float64
    infectiousness_gain_rate::Float64
    infectiousness_loss_rate::Float64
    self_recovery_rate::Float64
    detection_rate::Float64
    rel_detection_subclin::Float64
    tx_recovery_rate::Float64
    tx_relapse_rate::Float64
    tx_death_rate::Float64
    disease_mortality_clin_lowinf::Float64
    disease_mortality_clin_inf::Float64
    ageing_enabled::Bool
end

function make_parameters(
    model::RagonnetTBModel;
    beta::Real=1e-3,
    containment_child::Real=4.4,
    containment_adult::Real=2.0,
    clearance_rate::Real=0.02,
    breakdown_rate::Real=0.1,
    prop_infectious::Real=0.5,
    infectiousness_weights::NTuple{4,<:Real}=
    (0.2, 0.5, 0.4, 1.0),
    progression_child::Real=2.4,
    progression_5_14::Real=2.0,
    progression_15_64::Real=0.1,
    progression_65_plus::Real=2.4,
    clinical_progression_rate::Real=1.0,
    clinical_regression_rate::Real=1.0,
    infectiousness_gain_rate::Real=1.0,
    infectiousness_loss_rate::Real=1.0,
    self_recovery_rate::Real=0.4,
    detection_rate::Real=1.0,
    rel_detection_subclin::Real=0.0,
    tx_period::Real=0.5,
    tx_success_prop::Real=0.8,
    pct_neg_tx_death::Real=0.0,
    tx_recovery_rate::Union{Nothing,Real}=nothing,
    tx_relapse_rate::Union{Nothing,Real}=nothing,
    tx_death_rate::Union{Nothing,Real}=nothing,
    disease_mortality_clin_lowinf::Real=0.0,
    disease_mortality_clin_inf::Real=0.0,
    ageing_enabled::Bool=false,
)
    nages = state_layout(model).nages

    age_vector(f) = Float64[
        f(age) for age in 0:(nages-1)
    ]

    susceptibility_naive = ones(nages)
    susceptibility_contained = fill(0.2, nages)
    susceptibility_cleared = ones(nages)
    susceptibility_recovered = ones(nages)

    containment_rate = age_vector(
        age -> age < 15 ?
               containment_child : containment_adult
    )

    progression_rate = age_vector(age ->
        age < 5 ? progression_child :
        age < 15 ? progression_5_14 :
        age < 65 ? progression_15_64 :
        progression_65_plus
    )

    tx_rate = 1 / Float64(tx_period)
    failure = 1 - Float64(tx_success_prop)

    recovery = isnothing(tx_recovery_rate) ?
               Float64(tx_success_prop) * tx_rate :
               Float64(tx_recovery_rate)

    death = isnothing(tx_death_rate) ?
            failure * Float64(pct_neg_tx_death) * tx_rate :
            Float64(tx_death_rate)

    relapse = isnothing(tx_relapse_rate) ?
              failure * (1 - Float64(pct_neg_tx_death)) * tx_rate :
              Float64(tx_relapse_rate)

    return RagonnetTBParameters(
        copy(model.contact),
        Float64(beta),
        susceptibility_naive,
        susceptibility_contained,
        susceptibility_cleared,
        susceptibility_recovered,
        containment_rate,
        progression_rate,
        Float64.(infectiousness_weights),
        Float64(clearance_rate),
        Float64(breakdown_rate),
        Float64(prop_infectious),
        Float64(clinical_progression_rate),
        Float64(clinical_regression_rate),
        Float64(infectiousness_gain_rate),
        Float64(infectiousness_loss_rate),
        Float64(self_recovery_rate),
        Float64(detection_rate),
        Float64(rel_detection_subclin),
        recovery,
        relapse,
        death,
        Float64(disease_mortality_clin_lowinf),
        Float64(disease_mortality_clin_inf),
        ageing_enabled,
    )
end

make_parameters(
    model::RagonnetTBModel,
    draw::NamedTuple,
) = make_parameters(model; draw...)
