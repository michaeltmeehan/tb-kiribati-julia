# TB Kiribati Julia

A lightweight Julia implementation of an age-structured tuberculosis transmission model for Kiribati.

Phase 1 provides:

- 96 one-year age groups
- 10 epidemiological compartments
- age-specific susceptibility and progression
- age-structured contact mixing
- tuberculosis natural-history transitions
- treatment and disease mortality
- age-resolved cumulative epidemiological flows
- deterministic simulation via SciML

Phase 2 adds synthetic demographic coupling:

- background mortality
- continuous ageing from ages `0` through `94`
- an open-ended `95+` age group
- births into `MtbNaive` at age `0`
- proportional net migration
- annual stepwise demographic schedules

The demographic schedules in this phase are synthetic on purpose. WPP 2024 integration is deferred to Phase 3.

## Phase 3 WPP demography

Phase 3 replaces the synthetic demographic inputs with a Kiribati-specific extract from UN World Population Prospects 2024, validated against the `agepi` R workflow.

### Source data

The Julia loader in `src/wpp_demography.jl` reads compact CSV exports under `data/wpp2024/` that were generated from these WPP datasets:

- `popprojAge1dt` for population by age and year
- `mx1dt` for age-specific mortality rates
- `percentASFR1dt` and `tfrproj1dt` for fertility weights and TFR
- `migprojAge1dt` for age-specific net migration counts

### Conventions

- Population values are exported in persons, not thousands.
- Mortality is used as an annual hazard, matching the `agepi` WPP workflow.
- `fertility_mode = :agepi_compatible` preserves the agepi-style age-specific fertility reconstruction, while `fertility_mode = :wpp` uses the published annual births series from WPP.
- Net migration is treated as an annual count and then applied proportionally across TB compartments.
- The open-ended `95+` age group aggregates all ages `95` and above for counts, while mortality rates use the WPP `95+` rate without summing rates across older ages.
- WPP population values are end-of-calendar-year quantities; year-specific rates and counts apply to the calendar year itself using a stepwise annual schedule.
- The model remains sex-aggregated; fertility uses the `fertility_exposure_fraction = 0.5` convention from `agepi`.

### Reference files

The directory `validation/reference/agepi_kiribati/` contains the exported `agepi` comparison files:

- `population_2025.csv`
- `mortality_2025_2030.csv`
- `fertility_2025_2030.csv`
- `migration_2025_2030.csv`
- `trajectory_quarterly.csv`
- `metadata.csv`

### Scripts

- `data-raw/export_wpp_kiribati.R` exports the small WPP subset used by the Julia runtime.
- `validation/export_agepi_demography.R` exports the agepi reference files for numerical comparison.
- `scripts/run_kiribati_wpp.jl` runs the full Kiribati TB simulation with the WPP-backed demographic schedule.
- `scripts/compare_fertility_modes.jl` compares the `:agepi_compatible` and `:wpp` fertility modes side by side.

## Run

Phase 1 model:

```julia
julia --project=. scripts/run_model.jl
```

Phase 2 demographic model:

```julia
julia --project=. scripts/run_demographic_model.jl
```

Phase 3 WPP-backed model:

```julia
julia --project=. scripts/run_kiribati_wpp.jl
```

Compare fertility modes:

```bash
julia --project=. scripts/compare_fertility_modes.jl
```

## Tests

```julia
julia --project=. test/runtests.jl
```
