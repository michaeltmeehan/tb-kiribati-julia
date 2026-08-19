if (!requireNamespace("agepi", quietly = TRUE)) {
  stop("agepi must be installed to export the reference files.", call. = FALSE)
}
if (!requireNamespace("wpp2024", quietly = TRUE)) {
  stop("wpp2024 must be installed to export the reference files.", call. = FALSE)
}

library(agepi)
library(wpp2024)

country <- "Kiribati"
years <- 2025:2030
save_dir <- file.path("validation", "reference", "agepi_kiribati")
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

age_structure <- wpp_age_structure_1year(max_age = 95)
age_names <- age_structure$age_groups
named_age_vec <- function(x) {
  stats::setNames(as.numeric(x), age_names)
}
demography <- demographic_process_from_wpp(
  country = country,
  years = years,
  age_structure = age_structure,
  migration = TRUE,
  fertility_exposure_fraction = 0.5
)

population_2025 <- demography_population_vector(demography$population, time = 2025)
write.csv(
  data.frame(
    age = age_structure$age_groups,
    population = as.numeric(population_2025),
    stringsAsFactors = FALSE
  ),
  file.path(save_dir, "population_2025.csv"),
  row.names = FALSE,
  quote = FALSE
)
write.csv(demography$schedules$mortality$data, file.path(save_dir, "mortality_2025_2030.csv"), row.names = FALSE, quote = FALSE)
write.csv(demography$schedules$fertility$data, file.path(save_dir, "fertility_2025_2030.csv"), row.names = FALSE, quote = FALSE)
write.csv(demography$schedules$migration$data, file.path(save_dir, "migration_2025_2030.csv"), row.names = FALSE, quote = FALSE)

write.csv(
  data.frame(
    key = c(
      "country",
      "years",
      "age_structure",
      "migration",
      "fertility_exposure_fraction",
      "population_timing",
      "vital_rate_timing",
      "population_units",
      "mortality_convention",
      "fertility_convention",
      "migration_convention"
    ),
    value = c(
      country,
      "2025:2030",
      "0:95+",
      "TRUE",
      "0.5",
      "end-of-calendar-year",
      "calendar-year",
      "persons",
      "annual_hazard",
      "births_per_female_person_year",
      "count"
    ),
    stringsAsFactors = FALSE
  ),
  file.path(save_dir, "metadata.csv"),
  row.names = FALSE,
  quote = FALSE
)

age_lower <- age_structure$lower_bounds
tb_model <- CompartmentModel(
  compartments = c(
    "M.tb",
    "Contained",
    "Cleared",
    "Recovered",
    "Incipient",
    "Sub.clin.lowinf",
    "Sub.clin.inf",
    "Clin.lowinf",
    "Clin.inf",
    "Treatment"
  ),
  infection_transitions = data.frame(
    from = c("M.tb", "Contained", "Cleared", "Recovered"),
    to = c("Incipient", "Incipient", "Incipient", "Incipient"),
    susceptibility = I(list(
      named_age_vec(ifelse(age_lower < 15, 0.5, 1.0)),
      named_age_vec(rep(0.2, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower)))
    )),
    stringsAsFactors = FALSE
  ),
  infectious_compartments = c("Sub.clin.lowinf", "Sub.clin.inf", "Clin.lowinf", "Clin.inf"),
  infectiousness_weights = I(list(
    named_age_vec(ifelse(age_lower < 15, 0, 0.2)),
    named_age_vec(ifelse(age_lower < 15, 0, 0.5)),
    named_age_vec(ifelse(age_lower < 15, 0, 0.4)),
    named_age_vec(rep(1, length(age_lower)))
  )),
  beta = 1e-3,
  transitions = data.frame(
    name = c(
      "containment",
      "clearance",
      "breakdown",
      "progression.lowinf",
      "progression.inf",
      "clin.progression.lowinf",
      "clin.progression.inf",
      "clin.regression.lowinf",
      "clin.regression.inf",
      "infectious.gain.sub",
      "infectious.gain.clin",
      "infectiousness.loss.sub",
      "infectiousness.loss.clin",
      "self.recovery.lowinf",
      "self.recovery.inf",
      "detection.sub.lowinf",
      "detection.sub.inf",
      "detection.clin.lowinf",
      "detection.clin.inf",
      "tx.recovery",
      "relapse"
    ),
    from = c(
      "Incipient",
      "Contained",
      "Contained",
      "Incipient",
      "Incipient",
      "Sub.clin.lowinf",
      "Sub.clin.inf",
      "Clin.lowinf",
      "Clin.inf",
      "Sub.clin.lowinf",
      "Clin.lowinf",
      "Sub.clin.inf",
      "Clin.inf",
      "Sub.clin.lowinf",
      "Sub.clin.inf",
      "Sub.clin.lowinf",
      "Sub.clin.inf",
      "Clin.lowinf",
      "Clin.inf",
      "Treatment",
      "Treatment"
    ),
    to = c(
      "Contained",
      "Cleared",
      "Incipient",
      "Sub.clin.lowinf",
      "Sub.clin.inf",
      "Clin.lowinf",
      "Clin.inf",
      "Sub.clin.lowinf",
      "Sub.clin.inf",
      "Sub.clin.inf",
      "Clin.inf",
      "Sub.clin.lowinf",
      "Clin.lowinf",
      "Recovered",
      "Recovered",
      "Treatment",
      "Treatment",
      "Treatment",
      "Treatment",
      "Recovered",
      "Sub.clin.lowinf"
    ),
    rate = I(list(
      named_age_vec(ifelse(age_lower < 15, 4.4, 2.0)),
      named_age_vec(rep(0.02, length(age_lower))),
      named_age_vec(rep(0.1, length(age_lower))),
      named_age_vec(0.5 * ifelse(age_lower < 5, 2.4, ifelse(age_lower < 15, 2.0, ifelse(age_lower < 65, 0.1, 2.4)))),
      named_age_vec(0.5 * ifelse(age_lower < 5, 2.4, ifelse(age_lower < 15, 2.0, ifelse(age_lower < 65, 0.1, 2.4)))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(0.4, length(age_lower))),
      named_age_vec(rep(0.4, length(age_lower))),
      named_age_vec(rep(0.0, length(age_lower))),
      named_age_vec(rep(0.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.0, length(age_lower))),
      named_age_vec(rep(1.6, length(age_lower))),
      named_age_vec(rep(0.24, length(age_lower)))
    )),
    stringsAsFactors = FALSE
  ),
  outflows = data.frame(
    from = c("Clin.lowinf", "Clin.inf"),
    rate = c(0.025, 0.4),
    stringsAsFactors = FALSE
  ),
  birth_compartment = c("M.tb"),
  migration_compartment = c("M.tb")
)

initial_state <- data.frame(
  compartment = rep(tb_model$compartments, each = age_structure$n_age_groups),
  age_group = rep(age_structure$age_groups, times = length(tb_model$compartments)),
  value = c(
    0.99 * population_2025,
    rep(0, age_structure$n_age_groups),
    rep(0, age_structure$n_age_groups),
    rep(0, age_structure$n_age_groups),
    rep(0, age_structure$n_age_groups),
    rep(0, age_structure$n_age_groups),
    rep(0, age_structure$n_age_groups),
    rep(0, age_structure$n_age_groups),
    0.01 * population_2025,
    rep(0, age_structure$n_age_groups)
  ),
  stringsAsFactors = FALSE
)

tb_output <- simulate_deterministic(
  initial_state = initial_state,
  times = seq(2025, 2030, by = 0.05),
  model = tb_model,
  age_structure = age_structure,
  contact_matrix = diag(age_structure$n_age_groups),
  demographic_process = demography$demographic_process,
  time_policy = "step",
  migration_policy = "proportional",
  method = "euler"
)

trajectory <- tb_output
trajectory$time <- as.numeric(as.character(trajectory$time))
trajectory_quarterly <- trajectory[abs(round(trajectory$time * 4) - trajectory$time * 4) < 1e-8, ]
write.csv(trajectory_quarterly, file.path(save_dir, "trajectory_quarterly.csv"), row.names = FALSE, quote = FALSE)
