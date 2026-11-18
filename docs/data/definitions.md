## Principal derived variables

| Output                                | Source variables                                             | Transformation                                                   |
| ------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| National notifications                | `c_newinc`                                                   | Kiribati records with reported values                            |
| Age-specific notifications            | `newrel_m*`, `newrel_f*`                                     | Sum male and female observations within reported age groups      |
| Recurrent notifications, through 2012 | `ret_rel`                                                    | Reported relapse count                                           |
| Recurrent notifications, 2013–2024    | `ret_rel_labconf`, `ret_rel_clindx`, `ret_rel_ep`            | Sum reported recurrence categories                               |
| Pulmonary notifications, 2013–2024    | `new_labconf`, `new_clindx` and corresponding relapse fields | Retain laboratory and clinical categories separately             |
| Historical pulmonary notifications    | `new_sp`, `new_sn`                                           | Sum historical smear-positive and smear-negative notifications   |
| Historical pulmonary fraction         | `new_sp`, `new_sn`, `new_ep`                                 | Pooled pulmonary / total notifications, 2007–2012                |
| Population                            | `pop`, `popM`, `popF`                                        | Convert thousands to persons                                     |
| Age-specific fertility                | `tfr`, `pasfr`                                               | `asfr = tfr × pasfr / 100`                                       |
| Migration                             | `mig`                                                        | Convert thousands to persons                                     |
| Contact matrix                        | Prem 16×16                                                   | Expand contacts uniformly across source ages                     |
| Reciprocal contacts                   | Prem matrix, Kiribati population                             | Population-weighted reciprocity correction at 16-band resolution |

WHO national burden estimates retain their original field names. The suffixes `_lo` and `_hi` denote the source-reported uncertainty bounds; their statistical interpretation should follow WHO's documentation.

## Temporal conventions

WPP2024 population values are end-of-year quantities. Annual notification rates use the average of the population values for years \(t-1\) and \(t\) as an approximation to midyear exposure.

## Important distinctions

- A reported TB notification is not equivalent to a WHO-estimated incident TB case.
- A WHO case detection estimate is not a directly observed detection probability.
- Treatment-cohort deaths are not equivalent to population-level TB mortality.
- Historical smear classifications are not equivalent to modern laboratory-confirmed and clinically diagnosed pulmonary classifications.
- Population-adjusted reciprocity is an optional mixing assumption, not a correction to the original Prem source.
