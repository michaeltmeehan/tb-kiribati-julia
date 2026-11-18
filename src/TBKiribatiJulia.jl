module TBKiribatiJulia

using LinearAlgebra
using DifferentialEquations

include("data/contacts.jl")
include("data/notifications.jl")
include("data/surveillance.jl")

include("models/abstract.jl")

include("demography/data.jl")
include("demography/operators.jl")
include("demography/equilibrium.jl")
include("demography/projected.jl")

include("simulation/context.jl")

include("models/Dodd/model.jl")
include("models/Dodd/parameters.jl")
include("models/Dodd/dynamics.jl")
include("models/Dodd/initial_conditions.jl")
include("models/Dodd/priors.jl")

# Extended Dodd model
include("models/DoddExtended/model.jl")
include("models/DoddExtended/presentation.jl")
include("models/DoddExtended/parameters.jl")
include("models/DoddExtended/dynamics.jl")
include("models/DoddExtended/initial_conditions.jl")

include("simulation/simulate.jl")
include("simulation/equilibrium.jl")

include("outputs/counters.jl")
include("outputs/aggregation.jl")

include("models/DoddExtended/outputs.jl")

include("models/Dodd/outputs.jl")

include("inference/sampling.jl")

include("simulation/ensemble.jl")
include("outputs/summary.jl")

include("inference/likelihoods.jl")
include("inference/treatment_deaths.jl")
include("inference/recurrence.jl")
include("inference/calibration.jl")
include("inference/optimisation.jl")
include("inference/profiles.jl")
include("inference/model_comparison.jl")
include("inference/diagnostics.jl")
include("inference/annual_effects.jl")

include("analysis/sensitivity.jl")
include("visualisation/sensitivity.jl")

include("visualisation/calibration.jl")
include("visualisation/annual_effects.jl")
include("visualisation/profiles.jl")

include("inference/reporting.jl")

include("models/Ragonnet/model.jl")
include("models/Ragonnet/initial_conditions.jl")


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
export quadrature_convergence
export posterior_annual_effects
export plot_annual_effects
export AbstractCalibrationFit, MLEFit, fit_mle
export profile_likelihood, profile_likelihoods
export compare_calibrations, likelihood_ratio
export plot_profile_likelihood
export load_annual_counts,
    load_site_notifications,
    load_treatment_outcomes,
    load_historical_notification_types,
    load_kiribati_surveillance
export combine_notification_observations
export TreatmentDeathObservation,
    treatment_death_probability,
    treatment_death_loglikelihood,
    treatment_death_fit_rows
export RecurrentTBObservation,
    recurrence_loglikelihood
export DoddExtendedTBModel, DoddExtendedTBParameters
export extended_presentation, load_kiribati_presentation
export extended_fixed_parameters, extend_initial_state
export RagonnetTBModel, RagonnetTBParameters

end