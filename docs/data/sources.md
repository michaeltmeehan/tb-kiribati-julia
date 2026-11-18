# Data sources and provenance

## WHO Global TB Report 2026

Source: [GTB-TME/gtbreport2026](https://github.com/GTB-TME/gtbreport2026)

Export date: 25 August 2026.

### Reported data

| Local file                      | Original WHO export                  |
| ------------------------------- | ------------------------------------ |
| `reported/notifications.csv`    | `TB_notifications_2026-08-25.csv`    |
| `reported/outcomes.csv`         | `TB_outcomes_2026-08-25.csv`         |
| `reported/outcomes_age_sex.csv` | `TB_outcomes_age_sex_2026-08-25.csv` |

### Estimated data

| Local file                           | Original WHO export                            |
| ------------------------------------ | ---------------------------------------------- |
| `estimates/burden_countries.csv`     | `TB_burden_estimates_countries_2026-08-25.csv` |
| `estimates/burden_age_sex.csv`       | `TB_burden_estimates_age_sex_2026-08-25.csv`   |
| `estimates/preventive_treatment.csv` | `LTBI_estimates_2026-08-25.csv`                |

Local paths are relative to `data/raw/who/2026/`. The original downloads cover countries beyond Kiribati; extraction occurs during processing.

The national burden export contains estimated incidence, mortality and case detection, with uncertainty intervals. The age-specific burden export contains estimates for 2025.

## World Population Prospects 2024

Source: [PPgp/wpp2024](https://github.com/PPgp/wpp2024)

Country code: 296 (Kiribati).

The `extract_wpp2024.R` script extracts population, fertility, mortality, migration and supporting demographic totals.

Record the installed R package version in `data/raw/wpp2024/metadata.csv`.

## Prem contact estimates

Source: Prem et al. (2021), accessed via the R `contactdata` package.

Country: Solomon Islands.

Matrix: all locations, all geographic settings, 16 age groups.

The original matrix is retained under `data/raw/prem/`. The processing script expands this matrix into single-year ages.

The R package version, extraction date and source settings are recorded in `data/raw/prem/metadata.csv`.

## Provenance requirements

For future updates, record:

- Source URL and original filename.
- Download or extraction date.
- Source release or package version.
- Processing script.
- Changes in definitions or coverage relative to previous releases.
- File checksums where appropriate.

Do not silently replace the 2026 WHO exports with later releases.
