struct HistoricalTBParameters
    years::Vector{Int}
    beta::Vector{Float64}
    detection_rate::Vector{Float64}
    tx_success_prop::Vector{Float64}
    rel_detection_subclin::Vector{Float64}
    first_year::Int
    last_year::Int
end

mutable struct TBParams{M<:AbstractMatrix{Float64}, D}
    contact::M
    beta::Float64
    tx_period::Float64
    tx_success_prop::Float64
    pct_neg_tx_death::Float64
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
    historical::Union{Nothing,HistoricalTBParameters}
    simulation_regime::Symbol
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
    tx_period::Real = 0.5,
    tx_success_prop::Real = 0.8,
    pct_neg_tx_death::Real = 0.4,
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
    tx_recovery_rate::Union{Nothing,Real} = nothing,
    tx_relapse_rate::Union{Nothing,Real} = nothing,
    tx_death_rate::Union{Nothing,Real} = nothing,
    disease_mortality_clin_lowinf::Real = 0.025,
    disease_mortality_clin_inf::Real = 0.4,
    demography = nothing,
    historical::Union{Nothing,HistoricalTBParameters} = nothing,
    simulation_regime::Symbol = :historical,
    ageing_enabled::Bool = false,
)
    size(contact) == (NAGE, NAGE) || error("contact matrix must be 96×96")
    simulation_regime in (:historical, :stationary_burnin) || error("unsupported simulation_regime: $simulation_regime")

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
    tx_recovery = tx_recovery_rate === nothing ? Float64(tx_success_prop) * tx_rate : Float64(tx_recovery_rate)
    tx_failure_prop = 1 - Float64(tx_success_prop)
    tx_death = tx_death_rate === nothing ? tx_failure_prop * Float64(pct_neg_tx_death) * tx_rate : Float64(tx_death_rate)
    tx_relapse = tx_relapse_rate === nothing ? tx_failure_prop * (1 - Float64(pct_neg_tx_death)) * tx_rate : Float64(tx_relapse_rate)

    return TBParams(
        contact_matrix,
        Float64(beta),
        Float64(tx_period),
        Float64(tx_success_prop),
        Float64(pct_neg_tx_death),
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
        historical,
        simulation_regime,
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

function historical_parameter_schedule(
    years::AbstractVector{<:Integer};
    beta::AbstractVector{<:Real} = fill(1e-3, length(years)),
    detection_rate::AbstractVector{<:Real} = fill(1.0, length(years)),
    tx_success_prop::AbstractVector{<:Real} = fill(0.8, length(years)),
    rel_detection_subclin::AbstractVector{<:Real} = fill(0.0, length(years)),
)
    nyears = length(years)
    nyears > 0 || error("years must not be empty")

    yrs = Int[years...]
    for i in 2:nyears
        yrs[i] > yrs[i - 1] || error("years must be strictly increasing")
        yrs[i] == yrs[i - 1] + 1 || error("years must be consecutive annual values")
    end

    beta_vec = Float64[beta...]
    detection_vec = Float64[detection_rate...]
    tx_success_vec = Float64[tx_success_prop...]
    rel_detection_vec = Float64[rel_detection_subclin...]

    length(beta_vec) == nyears || error("beta must have one value per simulation year")
    length(detection_vec) == nyears || error("detection_rate must have one value per simulation year")
    length(tx_success_vec) == nyears || error("tx_success_prop must have one value per simulation year")
    length(rel_detection_vec) == nyears || error("rel_detection_subclin must have one value per simulation year")

    all(isfinite, beta_vec) || error("beta must contain only finite values")
    all(isfinite, detection_vec) || error("detection_rate must contain only finite values")
    all(isfinite, tx_success_vec) || error("tx_success_prop must contain only finite values")
    all(isfinite, rel_detection_vec) || error("rel_detection_subclin must contain only finite values")

    any(<(0.0), beta_vec) && error("beta must be non-negative")
    any(<(0.0), detection_vec) && error("detection_rate must be non-negative")
    any(<(0.0), tx_success_vec) && error("tx_success_prop must be non-negative")
    any(>(1.0), tx_success_vec) && error("tx_success_prop must not exceed 1")
    any(<(0.0), rel_detection_vec) && error("rel_detection_subclin must be non-negative")

    return HistoricalTBParameters(yrs, beta_vec, detection_vec, tx_success_vec, rel_detection_vec, yrs[1], yrs[end])
end

function demonstration_historical_parameters(years::AbstractVector{<:Integer})
    yrs = Int[years...]
    beta = fill(1e-3, length(yrs))
    detection = zeros(Float64, length(yrs))
    tx_success = fill(0.8, length(yrs))
    rel_detection = fill(0.0, length(yrs))

    @inbounds for (i, year) in pairs(yrs)
        detection[i] = if year < 1970
            0.0
        elseif year < 1990
            0.05
        elseif year < 2010
            0.10
        else
            0.15
        end
    end

    return historical_parameter_schedule(yrs; beta = beta, detection_rate = detection, tx_success_prop = tx_success, rel_detection_subclin = rel_detection)
end

@inline function historical_year_index(schedule::HistoricalTBParameters, t::Real)
    year = floor(Int, t)
    return clamp(year - schedule.first_year + 1, 1, length(schedule.years))
end

@inline function current_intervention_parameters(p::TBParams, t::Real)
    schedule = p.historical
    if p.simulation_regime === :stationary_burnin || schedule === nothing
        return p.beta, p.detection_rate, p.tx_recovery_rate, p.tx_relapse_rate, p.tx_death_rate, p.rel_detection_subclin
    end

    year_idx = historical_year_index(schedule, t)
    beta = schedule.beta[year_idx]
    detection_rate = schedule.detection_rate[year_idx]
    tx_rate = 1 / p.tx_period
    tx_success = schedule.tx_success_prop[year_idx]
    tx_failure_prop = 1 - tx_success
    tx_recovery = tx_success * tx_rate
    tx_death = tx_failure_prop * p.pct_neg_tx_death * tx_rate
    tx_relapse = tx_failure_prop * (1 - p.pct_neg_tx_death) * tx_rate
    rel_detection_subclin = schedule.rel_detection_subclin[year_idx]
    return beta, detection_rate, tx_recovery, tx_relapse, tx_death, rel_detection_subclin
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

function burnin_seed_state(
    population::AbstractVector{<:Real};
    naive_share::Real = 0.99,
    contained_share::Real = 0.005,
    cleared_share::Real = 0.0,
    recovered_share::Real = 0.0,
    incipient_share::Real = 0.0,
    subclin_low_share::Real = 0.0,
    subclin_inf_share::Real = 0.0,
    clin_low_share::Real = 0.0,
    clin_inf_share::Real = 0.005,
    treatment_share::Real = 0.0,
)
    shares = Float64[
        naive_share,
        contained_share,
        cleared_share,
        recovered_share,
        incipient_share,
        subclin_low_share,
        subclin_inf_share,
        clin_low_share,
        clin_inf_share,
        treatment_share,
    ]
    total = sum(shares)
    isapprox(total, 1.0; atol = 1e-12, rtol = 1e-12) || error("burnin seed shares must sum to 1")

    length(population) == NAGE || error("population vector must have length 96")
    u = zeros(Float64, NSTATE * NAGE)
    U = reshape(u, NSTATE, NAGE)
    @inbounds for a in 1:NAGE
        pop = Float64(population[a])
        for j in 1:NEPI
            U[j, a] = shares[j] * pop
        end
    end
    return u
end
