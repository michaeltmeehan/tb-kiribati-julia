if (!requireNamespace("agepi", quietly = TRUE)) {
  stop("agepi must be installed to export the contact matrix reference files.", call. = FALSE)
}
if (!requireNamespace("contactdata", quietly = TRUE)) {
  stop("contactdata must be installed to export the contact matrix reference files.", call. = FALSE)
}

library(agepi)
library(contactdata)

out_dir <- file.path("data", "contact")
ref_dir <- file.path("validation", "reference", "contact")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

country <- "Solomon Islands"
setting <- "all"
geographic_setting <- "all"
data_source <- "2020"
max_age <- 95

source_matrix <- contact_matrix(
  country = country,
  location = setting,
  geographic_setting = geographic_setting,
  data_source = data_source
)

source_age_bands <- data.frame(
  label = rownames(source_matrix),
  lower = c(seq(0, 70, by = 5), 75),
  upper = c(seq(4, 74, by = 5), Inf),
  stringsAsFactors = FALSE
)

write.table(
  source_matrix,
  file.path(out_dir, "prem_solomon_islands_all_source_matrix.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
write.csv(
  source_age_bands,
  file.path(out_dir, "prem_solomon_islands_all_agebands.csv"),
  row.names = FALSE,
  quote = FALSE
)

source_contacts <- load_contact_matrix_source(
  source = "prem",
  country = country,
  setting = setting
)
target_age_structure <- wpp_age_structure_1year(max_age = max_age)
adapted <- adapt_contact_matrix_to_age_structure(
  source_contacts,
  target_age_structure,
  method = "source_band"
)

write.table(
  adapted,
  file.path(ref_dir, "prem_solomon_islands_all_age96_agepi.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

write.csv(
  data.frame(
    key = c(
      "source_package",
      "source_dataset",
      "country",
      "setting",
      "geographic_setting",
      "data_source",
      "source_age_bands",
      "source_orientation",
      "source_units",
      "source_reciprocity",
      "source_population_weighting",
      "adaptation_method",
      "target_age_structure",
      "target_max_age",
      "target_orientation",
      "target_units",
      "target_reciprocity",
      "target_population_weighting"
    ),
    value = c(
      "contactdata",
      "Prem et al. 2017 / 2021 synthetic contact matrices",
      country,
      setting,
      geographic_setting,
      data_source,
      "00_05 | 05_10 | 10_15 | 15_20 | 20_25 | 25_30 | 30_35 | 35_40 | 40_45 | 45_50 | 50_55 | 55_60 | 60_65 | 65_70 | 70_75 | 75_80",
      "recipient_source",
      "contacts per person-year",
      "no",
      "no",
      "source_band",
      "0:95+",
      as.character(max_age),
      "recipient_source",
      "contacts per person-year",
      "no",
      "no"
    ),
    stringsAsFactors = FALSE
  ),
  file.path(ref_dir, "metadata.csv"),
  row.names = FALSE,
  quote = FALSE
)
