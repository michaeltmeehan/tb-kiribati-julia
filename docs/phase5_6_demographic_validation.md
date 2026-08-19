# Phase 5-6 Demographic Validation

## Goal

Validate the Kiribati demographic coupling against the WPP 2024 source data and preserve the existing `agepi`-compatible behavior while adding a WPP-consistent fertility mode.

## Findings

- The demographic mismatch is dominated by births, not ageing or migration.
- The WPP fertility extract exposes `percentASFR1dt` and `tfrproj1dt`, which `agepi` combines into age-specific fertility rates on the one-year age grid.
- That `agepi`-compatible reconstruction remains useful for regression compatibility, but it overstates births for this sex-aggregated TB model relative to WPP's published annual births.
- WPP also publishes annual births directly in `miscproj1dt`, which is the right quantity to feed into a sex-aggregated model that only needs total age-0 inflow.

## What Was Implemented

- `DemographicSchedule` now carries a `fertility_mode`.
- `fertility_mode = :agepi_compatible`
  - Reconstructs births from age-specific fertility rates and the current age distribution.
  - Preserves the existing Phase 3 / Phase 4 regression path.
- `fertility_mode = :wpp`
  - Uses the published annual births series directly.
  - Keeps the annual demographic accounting consistent with the WPP vital counts.
- `load_kiribati_wpp_data` and `wpp_kiribati_demographic_schedule` now accept `fertility_mode`.

## Validation Summary

- `:agepi_compatible` still matches the existing WPP-backed reference outputs used by the earlier phases.
- `:wpp` matches the published WPP births series for Kiribati for 2025-2030.
- Annual demographic balance closes under both modes.
- Age-0 balance remains correct under the WPP mode.

## Notes

- The TB transmission equations were not changed.
- The solver remains on the fast annual-step `Vern7()` path.
- The WPP fertility extract is still retained for diagnostics and compatibility checks.
