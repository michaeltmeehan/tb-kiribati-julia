if (!requireNamespace("wpp2024", quietly = TRUE)) {
  stop("wpp2024 must be installed to export the Kiribati subset.", call. = FALSE)
}

library(wpp2024)

country <- "Kiribati"
years <- 1950:2030
historical_years <- 1950:2023
projection_years <- 2024:2030
out_dir <- file.path("data", "wpp2024")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

data(popAge1dt)
data(popprojAge1dt)
data(percentASFR1dt)
data(tfr1dt)
data(mx1dt)
data(mig1dt)
data(migprojAge1dt)
data(misc1dt)
data(miscproj1dt)

population <- rbind(
  subset(popAge1dt, name == country & year %in% historical_years, select = c("name", "year", "age", "pop")),
  subset(popprojAge1dt, name == country & year %in% projection_years, select = c("name", "year", "age", "pop"))
)
population$year <- as.integer(population$year)
population$pop <- as.numeric(population$pop) * 1000
population <- population[order(population$year, population$age), ]
write.csv(population, file.path(out_dir, "population.csv"), row.names = FALSE, quote = FALSE)

mortality <- subset(mx1dt, name == country & year %in% years, select = c("name", "year", "age", "mxB"))
mortality$year <- as.integer(mortality$year)
mortality$mxB <- as.numeric(mortality$mxB)
mortality <- mortality[order(mortality$year, mortality$age), ]
write.csv(mortality, file.path(out_dir, "mortality.csv"), row.names = FALSE, quote = FALSE)

fertility_weights <- subset(percentASFR1dt, name == country & year %in% years, select = c("year", "age", "pasfr"))
fertility_weights$year <- as.integer(fertility_weights$year)
fertility_weights$age <- as.integer(fertility_weights$age)
fertility_weights$pasfr <- as.numeric(fertility_weights$pasfr)
tfr <- subset(tfr1dt, name == country & year %in% years, select = c("year", "tfr"))
tfr$year <- as.integer(tfr$year)
tfr$tfr <- as.numeric(tfr$tfr)
fertility <- merge(fertility_weights, tfr, by = "year", all.x = TRUE, sort = FALSE)
fertility <- fertility[order(fertility$year, fertility$age), ]
write.csv(fertility, file.path(out_dir, "fertility.csv"), row.names = FALSE, quote = FALSE)

template_migration <- subset(migprojAge1dt, name == country & year == 2024, select = c("age", "mig"))
template_migration$age <- as.integer(template_migration$age)
template_migration$mig <- as.numeric(template_migration$mig)
template_total <- sum(template_migration$mig)
if (template_total == 0) {
  template_migration$weight <- 0
} else {
  template_migration$weight <- template_migration$mig / template_total
}

historical_migration_total <- subset(mig1dt, name == country & year %in% historical_years, select = c("name", "year", "mig"))
historical_migration_total$year <- as.integer(historical_migration_total$year)
historical_migration_total$mig <- as.numeric(historical_migration_total$mig)
historical_migration <- merge(as.data.frame(historical_migration_total), as.data.frame(template_migration[, c("age", "weight")]), by = NULL)
historical_migration$mig <- historical_migration$mig * historical_migration$weight
historical_migration$name <- country
historical_migration <- historical_migration[, c("name", "year", "age", "mig")]

projection_migration <- subset(migprojAge1dt, name == country & year %in% projection_years, select = c("name", "year", "age", "mig"))
projection_migration$year <- as.integer(projection_migration$year)
projection_migration$age <- as.integer(projection_migration$age)
projection_migration$mig <- as.numeric(projection_migration$mig)

migration <- rbind(historical_migration, projection_migration)
migration$mig <- as.numeric(migration$mig) * 1000
migration <- migration[order(migration$year, migration$age), ]
write.csv(migration, file.path(out_dir, "migration.csv"), row.names = FALSE, quote = FALSE)

births <- rbind(
  as.data.frame(subset(misc1dt, name == country & year %in% historical_years, select = c("year", "births"))),
  as.data.frame(subset(miscproj1dt, name == country & year %in% projection_years, select = c("year", "births")))
)
births$year <- as.integer(births$year)
births$births <- as.numeric(births$births) * 1000
births <- births[order(births$year), ]
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
      "1950:2030",
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
