mutable struct TBParams{M<:AbstractMatrix{Float64}, D}
    contact::M
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
    demography::D
    ageing_enabled::Bool
    tmp_q::Vector{Float64}
    tmp_foi::Vector{Float64}
    tmp_age_pop::Vector{Float64}
end

function default_contact_matrix()
    Matrix{Float64}(I, NAGE, NAGE)
end

function default_population()
    fill(1000.0, NAGE)
end

function _age_grid()
    0:(NAGE - 1)
end

function _age_vector(f::Function)
    ages = _age_grid()
    out = Vector{Float64}(undef, NAGE)
    @inbounds for i in 1:NAGE
        out[i] = f(ages[i])
    end
    return out
end

function make_parameters(
    contact::AbstractMatrix{<:Real};
    beta::Real = 1e-3,
    containment_child::Real = 4.4,
    containment_adult::Real = 2.0,
    clearance_rate::Real = 0.02,
    breakdown_rate::Real = 0.1,
    prop_infectious::Real = 0.5,
    progression_child::Real = 2.4,
    progression_5_14::Real = 2.0,
    progression_15_64::Real = 0.1,
    progression_65_plus::Real = 2.4,
    clinical_progression_rate::Real = 1.0,
    clinical_regression_rate::Real = 1.0,
    infectiousness_gain_rate::Real = 1.0,
    infectiousness_loss_rate::Real = 1.0,
    self_recovery_rate::Real = 0.4,
    detection_rate::Real = 1.0,
    rel_detection_subclin::Real = 0.0,
    tx_period::Real = 0.5,
    tx_success_prop::Real = 0.8,
    pct_neg_tx_death::Real = 0.4,
    tx_recovery_rate::Union{Nothing,Real} = nothing,
    tx_relapse_rate::Union{Nothing,Real} = nothing,
    tx_death_rate::Union{Nothing,Real} = nothing,
    disease_mortality_clin_lowinf::Real = 0.025,
    disease_mortality_clin_inf::Real = 0.4,
    demography = nothing,
    ageing_enabled::Bool = false,
)
    size(contact) == (NAGE, NAGE) || error("contact matrix must be 96×96")

    contact_matrix = Matrix{Float64}(contact)
    susceptibility_naive = _age_vector(age -> age < 15 ? 0.5 : 1.0)
    susceptibility_contained = fill(0.2, NAGE)
    susceptibility_cleared = fill(1.0, NAGE)
    susceptibility_recovered = fill(1.0, NAGE)

    containment_rate = _age_vector(age -> age < 15 ? containment_child : containment_adult)
    progression_rate = _age_vector(age -> begin
        if age < 5
            progression_child
        elseif age < 15
            progression_5_14
        elseif age < 65
            progression_15_64
        else
            progression_65_plus
        end
    end)

    tx_rate = 1 / Float64(tx_period)
    tx_recovery = tx_recovery_rate === nothing ? tx_success_prop * tx_rate : Float64(tx_recovery_rate)
    tx_failure_prop = 1 - Float64(tx_success_prop)
    tx_death = tx_death_rate === nothing ? tx_failure_prop * Float64(pct_neg_tx_death) * tx_rate : Float64(tx_death_rate)
    tx_relapse = tx_relapse_rate === nothing ? tx_failure_prop * (1 - Float64(pct_neg_tx_death)) * tx_rate : Float64(tx_relapse_rate)

    return TBParams(
        contact_matrix,
        Float64(beta),
        susceptibility_naive,
        susceptibility_contained,
        susceptibility_cleared,
        susceptibility_recovered,
        containment_rate,
        progression_rate,
        (0.2, 0.5, 0.4, 1.0),
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
        tx_recovery,
        tx_relapse,
        tx_death,
        Float64(disease_mortality_clin_lowinf),
        Float64(disease_mortality_clin_inf),
        demography,
        ageing_enabled,
        zeros(Float64, NAGE),
        zeros(Float64, NAGE),
        zeros(Float64, NAGE),
    )
end

function make_default_parameters(; kwargs...)
    make_parameters(default_contact_matrix(); kwargs...)
end

function make_demographic_parameters(
    contact::AbstractMatrix{<:Real},
    demography::DemographicSchedule;
    ageing_enabled::Bool = true,
    kwargs...,
)
    make_parameters(contact; demography = demography, ageing_enabled = ageing_enabled, kwargs...)
end

function make_demographic_parameters(demography::DemographicSchedule; kwargs...)
    make_demographic_parameters(default_contact_matrix(), demography; kwargs...)
end

function initial_state(population::AbstractVector{<:Real})
    length(population) == NAGE || error("population vector must have length 96")
    u = zeros(Float64, NSTATE * NAGE)
    U = reshape(u, NSTATE, NAGE)
    @inbounds for a in 1:NAGE
        pop = Float64(population[a])
        U[MtbNaive, a] = 0.99 * pop
        U[ClinInf, a] = 0.01 * pop
    end
    return u
end
