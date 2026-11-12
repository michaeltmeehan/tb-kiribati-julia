module TBKiribatiJulia

using LinearAlgebra
using DifferentialEquations

include("data/contacts.jl")

include("models/abstract.jl")

include("demography/data.jl")
include("demography/operators.jl")
include("demography/equilibrium.jl")
include("demography/projected.jl")

include("simulation/context.jl")

include("models/dodd/model.jl")
include("models/dodd/parameters.jl")
include("models/dodd/dynamics.jl")
include("models/dodd/initial_conditions.jl")
include("models/dodd/priors.jl")

include("simulation/simulate.jl")
include("simulation/equilibrium.jl")

include("outputs/counters.jl")
include("outputs/aggregation.jl")

include("models/dodd/outputs.jl")

export load_contact_matrix,
    validate_contact_matrix,
    aggregate_contact_matrix
export AbstractTBModel
export StateLayout, state_layout, nstates, state_index
export DemographicData,
    load_demographic_data,
    get_population,
    fertility_rate,
    mortality_rate,
    migration_flow,
    EquilibriumDemography,
    apply_demography!,
    ProjectedDemography,
    target_population
export DoddTBModel, DoddTBParameters, DoddDetection
export dodd_priors
export make_parameters, initial_state, simulate
export equilibrate
export annual_events, aggregate_age_groups, model_outputs

end