module TBKiribatiJulia

using LinearAlgebra
using DifferentialEquations

include("data/contacts.jl")
include("data/notifications.jl")

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

include("inference/sampling.jl")

include("simulation/ensemble.jl")
include("outputs/summary.jl")

include("inference/likelihoods.jl")
include("inference/calibration.jl")

include("analysis/sensitivity.jl")
include("visualisation/sensitivity.jl")

include("inference/optimisation.jl")
include("inference/diagnostics.jl")
include("visualisation/calibration.jl")
include("inference/reporting.jl")

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
export priors
export make_parameters, initial_state, simulate
export equilibrate
export annual_events, aggregate_age_groups, model_outputs
export sample_prior
export simulate_from_draw
export simulate_ensemble
export summarise_ensemble
export sensitivity_data, sensitivity_correlations, sensitivity_prcc
export plot_parameter_pairs, plot_parameter_outcomes, plot_prcc_heatmap
export load_notification_observations
export nb2_loglikelihood, notification_loglikelihood
export calibration_predictions
export NB2Observation
export CalibrationProblem, model_draw, predict, notification_fit_rows
export calibration_loglikelihood, calibration_logprior, calibration_logposterior
export MAPFit, fit_map, calibration_diagnostics
export plot_calibration_national, plot_calibration_by_age, save_calibration
export AnnualRandomEffectNB2Observation
export observation_priors
export observation_variance
export observation_year_loglikelihood

end