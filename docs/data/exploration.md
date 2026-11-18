# Exploratory analysis of Kiribati TB data

## Scope

This document summarises the exploratory analyses of WHO Global
TB Report 2026 surveillance data and WPP2024 demographic data.

All numerical results are generated from processed CSVs.
Figures are available under `results/data_exploration/figures/`.

Observed notifications and WHO burden estimates are distinct
quantities and must not be interpreted interchangeably.


## National notifications

| Year | Notifications |
| --- | --- |
| 2013 | 407 |
| 2020 | 385 |
| 2023 | 565 |
| 2024 | 597 |
| 2025 | 541 |

Notification counts reflect both underlying TB disease
occurrence and ascertainment, including diagnostic and
reporting practices.

Age-specific notification counts are available for 2013–2025.
In 2024, childhood observations are reported as a combined
0–14 age group.


## Disease presentation

| Year | Laboratory-confirmed pulmonary | Clinically diagnosed pulmonary | Extrapulmonary |
| --- | --- | --- | --- |
| 2013 | 150 | 169 | 88 |
| 2020 | 200 | 45 | 140 |
| 2025 | 237 | 191 | 113 |

Pulmonary notifications comprise laboratory-confirmed and
clinically diagnosed pulmonary notifications.

Changes in these categories may reflect diagnostic or
reporting practices, not solely changes in disease burden.

Historical smear-positive and smear-negative classifications
are not directly equivalent to modern diagnostic categories.

The historical 2007–2012 pulmonary fractions describe
observed notifications, not underlying disease-onset
probabilities. Their figures show pooled Wilson intervals
and individual annual proportions.


## Recurrent notifications

| Year | Recurrent | Total | Percent recurrent |
| --- | --- | --- | --- |
| 2013 | 35 | 407 | 8.6 |
| 2020 | 14 | 385 | 3.6 |
| 2024 | 36 | 597 | 6.0 |

Recurrent notification definitions changed historically.
Trends should be interpreted in the context of those changes.

The recurrent proportion is conditional on reported total
notifications, rather than an independent incidence measure.


## WHO national burden estimates

| Year | Incident cases | Incident interval | TB deaths | Case detection (%) |
| --- | --- | --- | --- | --- |
| 2013 | 830 | 670–1000 | 45 | 49.0 |
| 2020 | 930 | 760–1100 | 46 | 41.0 |
| 2023 | 1100 | 900–1300 | 51 | 51.0 |
| 2024 | 1200 | 960–1400 | 52 | 51.0 |
| 2025 | 1100 | 880–1300 | 50 | 50.0 |

WHO incidence, mortality and case-detection estimates
are model-derived quantities, with reported uncertainty.

WHO case-detection estimates are not independent of
notifications and cannot automatically be added to a
notification likelihood as independent observations.

Age-specific WHO incidence estimates are not treated as
independent age-specific observations in this project.


## Demographic trends

| Year | Population |
| --- | --- |
| 1950 | 38228 |
| 2000 | 89376 |
| 2023 | 133527 |
| 2025 | 137468 |
| 2050 | 183463 |
| 2100 | 222638 |

Demographic estimates and projections come from WPP2024.
Population values represent end-of-year populations.

Annual notification rates use the average of consecutive
end-of-year populations to approximate midyear person-time.
This approximation should be reconsidered if exact
exposure estimates become available.


## Notification rates

| Year | Crude rate | Age-standardised rate |
| --- | --- | --- |
| 2013 | 360.2 | 369.8 |
| 2020 | 305.3 | 305.9 |
| 2023 | 426.3 | 426.3 |
| 2024 | 443.8 | 443.4 |
| 2025 | 396.4 | 394.5 |

Rates are per 100,000 population.

Direct age standardisation uses Kiribati's 2023
population distribution as the reference.

Seven consistent age groups are used:
0–14, 15–24, 25–34, 35–44, 45–54, 55–64 and 65+.
This avoids artificially splitting the combined
childhood observations in 2024.


## Demographic decomposition

Notification changes are decomposed into contributions from:

1. Total population size.
2. Population age composition.
3. Age-specific notification rates.

The decomposition averages marginal contributions across
all six factor orderings (three-factor Shapley method).

The contributions sum exactly to the observed change
in age-aggregated notifications.

| Period | Size | Age structure | Rates | Total |
| --- | --- | --- | --- | --- |
| 2013–2025 | 85.9 | 10.5 | 34.6 | 131.0 |
| 2023–2025 | 16.3 | 2.6 | -42.9 | -24.0 |

The rate contribution includes changes in underlying
disease incidence, case detection and reporting.

The decomposition uses age-specific notification totals.
These differ from independently reported national totals
in 2013 and 2015.


## Limitations and outstanding questions

- Surveillance definitions and reporting practices change.
- WHO burden estimates depend partly on reported notifications.
- Treatment-cohort mortality differs from population-level TB mortality.
- The Solomon Islands contact matrix is a proxy for Kiribati.
- The 2024 childhood observations cannot be disaggregated.
- Age-specific notification-rate changes cannot be attributed
  exclusively to epidemiological changes.

## Reproducing the analyses

From the Julia project environment, run:

```julia
include("scripts/data/explore_surveillance.jl")
include("scripts/data/explore_demography.jl")
include("scripts/data/decompose_notifications.jl")
include("scripts/data/document_exploration.jl")
```

All analyses consume processed data without modifying them.

