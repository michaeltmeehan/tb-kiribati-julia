
if (!requireNamespace("contactdata", quietly = TRUE)) {
  install.packages("contactdata")
}

root <- normalizePath(".", winslash = "/", mustWork = TRUE)
outdir <- file.path(root, "data", "raw", "prem")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# Prem et al. 2021 (labelled "2020" in contactdata)
contacts <- contactdata::contact_matrix(
  country = "Solomon Islands",
  location = "all",
  geographic_setting = "all",
  data_source = "2020"
)

stopifnot(
  identical(dim(contacts), c(16L, 16L)),
  all(is.finite(contacts)),
  all(contacts >= 0)
)

old_digits <- getOption("digits")
options(digits = 17)

write.table(
  contacts,
  file = file.path(outdir, "solomon_islands_all_16x16.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

options(digits = old_digits)

metadata <- data.frame(
  key = c(
    "source",
    "source_package",
    "package_version",
    "country",
    "location",
    "geographic_setting",
    "data_source",
    "extraction_date"
  ),
  value = c(
    "Prem et al. (2021)",
    "contactdata",
    as.character(packageVersion("contactdata")),
    "Solomon Islands",
    "all",
    "all",
    "2020",
    as.character(Sys.Date())
  )
)

write.csv(
  metadata,
  file.path(outdir, "metadata.csv"),
  row.names = FALSE
)

cat("Extracted Prem matrix:", dim(contacts), "\n")
