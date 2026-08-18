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

## Run

Phase 1 model:

```julia
julia --project=. scripts/run_model.jl
```

Phase 2 demographic model:

```julia
julia --project=. scripts/run_demographic_model.jl
```

## Tests

```julia
julia --project=. test/runtests.jl
```
