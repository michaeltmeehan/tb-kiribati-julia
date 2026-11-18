
if (!requireNamespace("wpp2024", quietly = TRUE)) {
  stop("Install wpp2024 from PPgp/wpp2024 first.")
}

root <- normalizePath(".", winslash = "/", mustWork = TRUE)
outdir <- file.path(root, "data", "raw", "wpp2024")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

country_code <- 296L  # Kiribati

load_wpp <- function(name) {
  e <- new.env(parent = baseenv())
  data(list = name, package = "wpp2024", envir = e)
  if (!exists(name, envir = e, inherits = FALSE)) {
    stop("Missing WPP2024 dataset: ", name)
  }
  as.data.frame(get(name, envir = e))
}

kiribati <- function(name) {
  x <- load_wpp(name)
  x <- x[x$country_code == country_code, , drop = FALSE]
  if (!nrow(x)) stop("No Kiribati data in ", name)
  x
}

write_raw <- function(x, name) {
  path <- file.path(outdir, paste0(name, ".csv"))
  write.csv(x, path, row.names = FALSE, na = "")
  message(name, ": ", nrow(x), " rows")
}

# --------------------------------------------------
# Population: historical + projected
# --------------------------------------------------

pop_hist <- kiribati("popAge1dt")
pop_proj <- kiribati("popprojAge1dt")

population <- rbind(
  pop_hist[pop_hist$year <= 2023,
           c("year", "age", "popM", "popF", "pop")],
  pop_proj[pop_proj$year >= 2024,
           c("year", "age", "popM", "popF", "pop")]
)

population <- population[
  order(population$year, population$age), ]
write_raw(population, "population")

# --------------------------------------------------
# Fertility: total fertility rate + age distribution
# --------------------------------------------------

tfr_hist <- kiribati("tfr1dt")
tfr_proj <- kiribati("tfrproj1dt")

tfr <- rbind(
  tfr_hist[tfr_hist$year <= 2023,
           c("year", "tfr")],
  tfr_proj[tfr_proj$year >= 2024,
           c("year", "tfr")]
)

pasfr <- kiribati("percentASFR1dt")
pasfr <- pasfr[, c("year", "age", "pasfr")]

fertility <- merge(pasfr, tfr, by = "year")
fertility <- fertility[
  order(fertility$year, fertility$age), ]

write_raw(fertility, "fertility")

# --------------------------------------------------
# Mortality: annual central death rates
# --------------------------------------------------

mortality <- kiribati("mx1dt")
mortality <- mortality[
  order(mortality$year, mortality$age),
  c("year", "age", "mxM", "mxF", "mxB")
]

write_raw(mortality, "mortality")

# --------------------------------------------------
# Total annual net migration
# --------------------------------------------------

migration <- kiribati("migration1dt")
migration <- migration[
  order(migration$year),
  c("year", "mig")
]

write_raw(migration, "migration")

# --------------------------------------------------
# Auxiliary totals for demographic validation
# --------------------------------------------------

misc_hist <- kiribati("misc1dt")
misc_proj <- kiribati("miscproj1dt")

fields <- c(
  "year", "births", "deaths",
  "PopChange", "cnmr"
)

vital_totals <- rbind(
  misc_hist[misc_hist$year <= 2023, fields],
  misc_proj[misc_proj$year >= 2024, fields]
)

vital_totals <- vital_totals[
  order(vital_totals$year), ]

write_raw(vital_totals, "vital_totals")

# --------------------------------------------------
# Metadata
# --------------------------------------------------

metadata <- data.frame(
  key = c(
    "source",
    "package_version",
    "country",
    "country_code",
    "population_units",
    "population_timing",
    "rate_timing",
    "migration_units",
    "fertility_distribution_units",
    "extraction_date"
  ),
  value = c(
    "UN WPP2024 via PPgp/wpp2024",
    as.character(packageVersion("wpp2024")),
    "Kiribati",
    "296",
    "thousands of persons",
    "31 December of labelled year",
    "calendar year",
    "thousands of persons per year",
    "percent of TFR across ages",
    as.character(Sys.Date())
  )
)

write_raw(metadata, "metadata")

message("WPP2024 extraction complete.")
