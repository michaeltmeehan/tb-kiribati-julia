
struct RagonnetTBParameters{T<:Real}
    contact::Matrix{Float64}
    beta::T

    susceptibility_naive::Vector{T}
    susceptibility_contained::Vector{T}
    susceptibility_cleared::Vector{T}
    susceptibility_recovered::Vector{T}

    containment_rate::Vector{T}
    progression_rate::Vector{T}
    infectiousness_weights::NTuple{4,T}

    clearance_rate::T
    breakdown_rate::T
    prop_infectious::T

    clinical_progression_rate::T
    clinical_regression_rate::T
    infectiousness_gain_rate::T
    infectiousness_loss_rate::T
    self_recovery_rate::T

    detection_rate::T
    rel_detection_subclin::T

    tx_recovery_rate::T
    tx_relapse_rate::T
    tx_death_rate::T

    disease_mortality_clin_lowinf::T
    disease_mortality_clin_inf::T

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
    numeric_inputs = (
        beta, containment_child, containment_adult,
        clearance_rate, breakdown_rate, prop_infectious,
        infectiousness_weights...,
        progression_child, progression_5_14,
        progression_15_64, progression_65_plus,
        clinical_progression_rate,
        clinical_regression_rate,
        infectiousness_gain_rate,
        infectiousness_loss_rate,
        self_recovery_rate, detection_rate,
        rel_detection_subclin,
        tx_period, tx_success_prop, pct_neg_tx_death,
        tx_recovery_rate, tx_relapse_rate, tx_death_rate,
        disease_mortality_clin_lowinf,
        disease_mortality_clin_inf,
    )

    T = foldl(
        promote_type,
        (typeof(x) for x in numeric_inputs if x isa Real);
        init=Float64,
    )

    nages = state_layout(model).nages

    age_vector(f) = T[f(a) for a in 0:(nages-1)]

    susceptibility_naive = ones(T, nages)
    susceptibility_contained = fill(T(0.2), nages)
    susceptibility_cleared = ones(T, nages)
    susceptibility_recovered = ones(T, nages)

    containment_rate = age_vector(
        age -> age < 15 ?
               containment_child : containment_adult
    )

    progression_rate = age_vector(
        age -> age < 5 ? progression_child :
               age < 15 ? progression_5_14 :
               age < 65 ? progression_15_64 :
               progression_65_plus
    )

    tx_rate = inv(T(tx_period))
    failure = one(T) - T(tx_success_prop)

    recovery = isnothing(tx_recovery_rate) ?
               T(tx_success_prop) * tx_rate :
               T(tx_recovery_rate)

    death = isnothing(tx_death_rate) ?
            failure * T(pct_neg_tx_death) * tx_rate :
            T(tx_death_rate)

    relapse = isnothing(tx_relapse_rate) ?
              failure * (one(T) - T(pct_neg_tx_death)) *
        tx_rate :
              T(tx_relapse_rate)

    return RagonnetTBParameters(
        copy(model.contact),
        T(beta),
        susceptibility_naive,
        susceptibility_contained,
        susceptibility_cleared,
        susceptibility_recovered,
        containment_rate,
        progression_rate,
        Tuple(T.(infectiousness_weights)),
        T(clearance_rate),
        T(breakdown_rate),
        T(prop_infectious),
        T(clinical_progression_rate),
        T(clinical_regression_rate),
        T(infectiousness_gain_rate),
        T(infectiousness_loss_rate),
        T(self_recovery_rate),
        T(detection_rate),
        T(rel_detection_subclin),
        recovery,
        relapse,
        death,
        T(disease_mortality_clin_lowinf),
        T(disease_mortality_clin_inf),
        ageing_enabled,
    )
end

make_parameters(
    model::RagonnetTBModel,
    draw::NamedTuple,
) = make_parameters(model; draw...)
