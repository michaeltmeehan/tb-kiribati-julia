# Data validation and known inconsistencies

## Surveillance

### National versus age-specific notifications

| Year | Age-specific total | National total | Difference |
| ---- | -----------------: | -------------: | ---------: |
| 2013 |                410 |            407 |         +3 |
| 2015 |                516 |            496 |        +20 |

These discrepancies occur in the WHO source fields and are not corrected during processing.

### Historical presentation, 2006

| Category | Sum of age records | Reported national total | Difference |
| -------- | -----------------: | ----------------------: | ---------: |
| `new_sp` |                125 |                     129 |         −4 |
| `new_sn` |                103 |                     117 |        −14 |

The 2007–2012 records are retained for analyses of historical age-specific presentation, excluding the inconsistent 2006 records.

### Treatment outcomes

The reconstructed main treatment cohorts balance exactly against the sum of recorded outcomes.

The 2024 cohort uses `nrr_tx`, whereas earlier years use `newrel`. The change in reporting definitions is preserved.

### Reporting changes

- Historical and modern pulmonary classification systems differ.
- The 2024 childhood notification observations include an aggregated 0–14 category.
- National and age-specific WHO incidence estimates may differ slightly because of estimation and rounding.

## Demography

The reconstructed WPP2024 inputs reproduce the previous population, fertility, mortality and migration arrays to numerical precision.

Maximum absolute differences:

| Input                         |   Difference |
| ----------------------------- | -----------: |
| Population (legacy thousands) | 4.44 × 10⁻¹⁶ |
| Fertility                     | 1.11 × 10⁻¹⁶ |
| Mortality                     |            0 |
| Migration                     |            0 |

The reconstruction does not establish that the original demographic assumptions are epidemiologically optimal; it verifies faithful reproduction.

## Contacts

The reconstructed Prem matrix reproduces the previous 96×96 matrix to numerical precision.

The reciprocity-corrected matrix satisfies population-weighted reciprocity at the original 16-age-group resolution.

For the fixed-2023 correction:

- Relative Frobenius difference from the original: 14.34%.
- Original mean row sum: 12.923.
- Corrected mean row sum: 13.779.

The original matrix remains available for sensitivity analysis.

## Validation policy

Discrepancies attributable to source data should be recorded, not silently adjusted.

Automated tests should distinguish structural errors, missing observations, inconsistent totals and scientifically recognised source discrepancies.
