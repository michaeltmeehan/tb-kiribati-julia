
module TBKiribatiJulia

using LinearAlgebra
using DifferentialEquations

# ============================================================
# Data
# ============================================================

include("data/contacts.jl")
include("data/notifications.jl")
include("data/surveillance.jl")

# ============================================================
# Model interfaces
# ============================================================

include("models/abstract.jl")

# ============================================================
# Demography
# ============================================================

include("demography/data.jl")
include("demography/equilibrium.jl")
include("demography/projected.jl")
include("demography/cohort.jl")
include("demography/operators.jl")
include("demography/growth_only.jl")

# ============================================================
# Shared numerical infrastructure and mixing
# ============================================================

include("simulation/context.jl")
include("mixing/model_mixing.jl")

# ============================================================
# Dodd
# ============================================================

include("models/Dodd/model.jl")
include("models/Dodd/parameters.jl")
include("models/Dodd/dynamics.jl")
include("models/Dodd/initial_conditions.jl")
include("models/Dodd/priors.jl")

# ============================================================
# DoddExtended
# ============================================================

include("models/DoddExtended/model.jl")
include("models/DoddExtended/presentation.jl")
include("models/DoddExtended/parameters.jl")
include("models/DoddExtended/dynamics.jl")
include("models/DoddExtended/initial_conditions.jl")

# ============================================================
# Ragonnet
# ============================================================

include("models/Ragonnet/model.jl")
include("models/Ragonnet/initial_conditions.jl")

# ============================================================
# Simulation
# ============================================================

include("simulation/simulate.jl")
include("simulation/equilibrium.jl")

# ============================================================
# Epidemiological outputs
# ============================================================

include("outputs/counters.jl")
include("outputs/aggregation.jl")
include("outputs/extraction.jl")

include("models/Dodd/outputs.jl")
include("models/DoddExtended/outputs.jl")
include("models/Ragonnet/outputs.jl")

include("outputs/summary.jl")

# ============================================================
# Ensemble simulation
# ============================================================

include("simulation/ensemble.jl")

# ============================================================
# Projections
# ============================================================

include("projections/simulation.jl")
include("projections/summaries.jl")

# ============================================================
# Inference
# ============================================================

include("inference/sampling.jl")

# Observation distributions and likelihood primitives
include("inference/likelihoods.jl")

# Observation-specific likelihoods
include("inference/notifications.jl")
include("inference/treatment_deaths.jl")
include("inference/recurrence.jl")
include("inference/site.jl")

# Calibration and optimisation
include("inference/calibration.jl")
include("inference/optimisation.jl")

# Inference diagnostics and reporting
include("inference/profiles.jl")
include("inference/model_comparison.jl")
include("inference/diagnostics.jl")
include("inference/annual_effects.jl")
include("inference/reporting.jl")

# ============================================================
# Analysis and visualisation
# ============================================================

include("analysis/sensitivity.jl")

include("visualisation/sensitivity.jl")
include("visualisation/calibration.jl")
include("visualisation/annual_effects.jl")
include("visualisation/profiles.jl")
include("visualisation/projections.jl")
include("visualisation/priors.jl")

# ============================================================
# Exports
# ============================================================

# Data
export load_contact_matrix,
    validate_contact_matrix,
    aggregate_contact_matrix

export load_notification_observations

export load_annual_counts,
    load_site_notifications,
    load_treatment_outcomes,
    load_historical_notification_types,
    load_kiribati_surveillance

export combine_notification_observations

# Model interface
export AbstractTBModel
export StateLayout, state_layout, nstates, state_index
export state_dimension, validate_model_interface
export make_parameters, initial_state, priors

# Models
export DoddTBModel, DoddTBParameters, DoddDetection
export DoddExtendedTBModel, DoddExtendedTBParameters
export RagonnetTBModel, RagonnetTBParameters

export extended_presentation, load_kiribati_presentation
export extended_fixed_parameters

# Demography
export DemographicData,
    load_demographic_data,
    get_population,
    fertility_rate,
    mortality_rate,
    migration_flow,
    EquilibriumDemography,
    ProjectedDemography,
    CohortDemography,
    GrowthOnlyDemography,
    apply_demography!,
    target_population

export age_and_survive!, reconcile_population!
export demographic_mortality_year
export load_age_migration, cohort_migration

# Mixing
export normalise_model_contacts, adult_infectiousness

# Simulation
export simulate, simulate_from_draw
export PreparedSimulation, prepare_simulation
export equilibrate, equilibrium_parameter_check
export simulate_ensemble, summarise_ensemble

# Projections
export project_scenarios, summarise_projections, compare_projections

# Outputs
export annual_events, annual_windows
export aggregate_age_groups
export model_output, model_outputs

# Sampling and inference
export sample_prior

export nb2_loglikelihood, notification_loglikelihood
export NB2Observation

export CalibrationProblem, model_draw, predict,
    notification_fit_rows

export calibration_loglikelihood,
    calibration_logprior,
    calibration_logposterior

export MAPFit, fit_map, calibration_diagnostics

export AnnualRandomEffectNB2Observation
export observation_priors
export observation_variance
export observation_year_loglikelihood
export quadrature_convergence
export posterior_annual_effects

export AbstractCalibrationFit, MLEFit, fit_mle
export profile_likelihood, profile_likelihoods
export compare_calibrations, likelihood_ratio

export TreatmentDeathObservation,
    treatment_death_probability,
    treatment_death_loglikelihood,
    treatment_death_fit_rows

export RecurrentTBObservation,
    recurrence_loglikelihood

export SiteTBObservation, site_loglikelihood

# Analysis
export sensitivity_data,
    sensitivity_correlations,
    sensitivity_prcc

# Visualisation and reporting
export plot_parameter_pairs,
    plot_parameter_outcomes,
    plot_prcc_heatmap

export plot_calibration_national,
    plot_calibration_by_age,
    save_calibration

export plot_annual_effects
export plot_profile_likelihood

export plot_projections,
    plot_projections!,
    plot_projection_effects,
    plot_projection_effects!

export plot_prior_posterior

end # module TBKiribatiJulia
