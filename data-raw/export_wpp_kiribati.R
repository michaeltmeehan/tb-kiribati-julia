if (!requireNamespace("wpp2024", quietly = TRUE)) {
  stop("wpp2024 must be installed to export the Kiribati subset.", call. = FALSE)
}

library(wpp2024)

country <- "Kiribati"
years <- 2025:2030
out_dir <- file.path("data", "wpp2024")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

data(popprojAge1dt)
data(percentASFR1dt)
data(tfrproj1dt)
data(mx1dt)
data(migprojAge1dt)
data(miscproj1dt)

population <- subset(popprojAge1dt, name == country & year %in% years, select = c("name", "year", "age", "pop"))
population$year <- as.integer(population$year)
population$pop <- as.numeric(population$pop)
population$pop <- 1000 * population$pop
write.csv(population, file.path(out_dir, "population.csv"), row.names = FALSE, quote = FALSE)

mortality <- subset(mx1dt, name == country & year %in% years, select = c("name", "year", "age", "mxB"))
mortality$year <- as.integer(mortality$year)
mortality$mxB <- as.numeric(mortality$mxB)
write.csv(mortality, file.path(out_dir, "mortality.csv"), row.names = FALSE, quote = FALSE)

fertility_weights <- subset(percentASFR1dt, name == country & year %in% years, select = c("year", "age", "pasfr"))
fertility_weights$year <- as.integer(fertility_weights$year)
fertility_weights$age <- as.integer(fertility_weights$age)
fertility_weights$pasfr <- as.numeric(fertility_weights$pasfr)
tfr <- subset(tfrproj1dt, name == country & year %in% years, select = c("year", "tfr"))
tfr$year <- as.integer(tfr$year)
tfr$tfr <- as.numeric(tfr$tfr)
fertility <- merge(fertility_weights, tfr, by = "year", all.x = TRUE, sort = FALSE)
write.csv(fertility, file.path(out_dir, "fertility.csv"), row.names = FALSE, quote = FALSE)

migration <- subset(migprojAge1dt, name == country & year %in% years, select = c("name", "year", "age", "mig"))
migration$year <- as.integer(migration$year)
migration$mig <- as.numeric(migration$mig)
write.csv(migration, file.path(out_dir, "migration.csv"), row.names = FALSE, quote = FALSE)

births <- subset(miscproj1dt, name == country & year %in% years, select = c("year", "births"))
births$year <- as.integer(births$year)
births$births <- as.numeric(births$births) * 1000
write.csv(births, file.path(out_dir, "births.csv"), row.names = FALSE, quote = FALSE)

write.csv(
  data.frame(
    key = c(
      "country",
      "years",
      "population_units",
      "mortality_units",
      "fertility_units",
      "births_units",
      "migration_units",
      "population_timing",
      "vital_rate_timing",
      "open_age",
      "fertility_exposure_fraction"
    ),
    value = c(
      country,
      "2025:2030",
      "persons",
      "annual hazard / central death rate proxy",
      "births per female person-year",
      "persons per year",
      "persons per year",
      "end-of-calendar-year population",
      "calendar-year rates/counts",
      "95+",
      "0.5"
    ),
    stringsAsFactors = FALSE
  ),
  file.path(out_dir, "metadata.csv"),
  row.names = FALSE,
  quote = FALSE
)
