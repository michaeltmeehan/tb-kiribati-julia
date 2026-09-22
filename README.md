# TBKiribatiJulia

`TBKiribatiJulia` is a Julia implementation of an age-structured tuberculosis (TB) transmission model developed for studying TB epidemiology in Kiribati.

The package combines TB natural-history dynamics with age-specific contact patterns and demography. It supports both a fixed equilibrium demographic regime and time-varying demographic projections, allowing the epidemiological consequences of population ageing to be studied separately from population growth.

> **Status:** active research code. The core transmission model, demographic system, simulation interface, summary utilities, calibration workflow, and demographic comparison analysis are implemented. Parameter calibration remains provisional and should not yet be interpreted as a final fitted model of TB burden in Kiribati.

## Model overview

The model is a deterministic, age-structured compartmental model solved as a system of ordinary differential equations.

Current features include:

- 96 one-year age groups, representing ages 0 through 95+;
- age-specific population demography;
- age-specific contact mixing;
- TB infection, progression, containment, recovery, detection/treatment, and mortality dynamics;
- age-dependent progression parameters;
- differential infectiousness across infectious disease states;
- TB-related mortality;
- cumulative epidemiological outcomes for incidence and deaths;
- equilibrium and dynamic demographic simulation modes;
- utilities for summarising total and age-specific epidemiological outcomes;
- utilities for summarising demographic change;
- burn-in simulations for constructing epidemiologically consistent initial conditions;
- calibration scripts for matching model output to age-specific TB burden estimates.

The package is currently being used to investigate how projected changes in the **age structure** of the Kiribati population may affect future TB incidence and mortality.

## Scientific question

A principal analysis compares two projection scenarios from the same epidemiological state:

1. **Equilibrium demography** — the age structure is held under the model's equilibrium demographic regime.
2. **Dynamic demography** — age-specific demographic rates evolve according to the demographic projections used by the package.

The comparison is designed to isolate the effect of demographic ageing. Epidemiological comparisons therefore focus primarily on population-normalised outcomes such as incidence and mortality per 100,000 population rather than raw event counts, which would also reflect differences in total population growth.

## Installation

Clone the repository and activate the project environment:

```bash
git clone https://github.com/michaeltmeehan/tb-kiribati-julia.git
cd tb-kiribati-julia
julia --project=.
```

From the Julia REPL:

```julia
using Pkg
Pkg.instantiate()
```

The package can then be loaded with:

```julia
using TBKiribatiJulia
```

## Basic simulation workflow

A typical simulation has four steps:

1. construct model parameters;
2. construct or obtain an initial population state;
3. simulate the model;
4. summarise the solution.

For example:

```julia
using TBKiribatiJulia

params = make_parameters(
    CONTACT;
    beta = 0.75,
    progression_child = 3.0,
    progression_5_14 = 0.1,
    progression_15_64 = 0.25,
    progression_65_plus = 0.5,
    infectiousness_weights = (0.2, 0.5, 0.4, 1.0),
    pct_neg_tx_death = 0.4,
    disease_mortality_clin_lowinf = 0.025,
    disease_mortality_clin_inf = 0.4,
)

population0 = get_population(EQUILIBRIUM_YEAR)
u0 = seeded_initial_state(population0)

sol = simulate(
    params;
    tspan = (1800.0, 2025.0),
    u0 = u0,
    saveat = 1800.0:1.0:2025.0,
    demography = :equilibrium,
)
```

The resulting solution can be converted into epidemiological summaries with:

```julia
summary = simulation_summary(sol)
```

Current summary outputs include:

```julia
summary.years
summary.population
summary.population_by_age
summary.incidence
summary.incidence_by_age
summary.incidence_per_100k
summary.deaths
summary.deaths_by_age
summary.deaths_per_100k
```

Demographic summaries can be generated with:

```julia
demography = demographic_summary(summary)
```

which currently provides:

```julia
demography.years
demography.population
demography.median_age
demography.prop_under_5
demography.prop_65_plus
```

## Comparing equilibrium and dynamic demography

The main scientific comparison is implemented in `scripts/compare_demography.jl`.

The workflow first runs a common burn-in under equilibrium demography. Both projection scenarios then begin from **exactly the same epidemiological state**, ensuring that subsequent differences arise from their demographic assumptions rather than different initial conditions.

A simplified version is:

```julia
using TBKiribatiJulia

burnin_start = 1800.0
projection_start = 2025.0
projection_end = 2100.0

burnin_times = burnin_start:1.0:projection_start
projection_times = projection_start:1.0:projection_end

params = make_parameters(
    CONTACT;
    beta = 0.75,
    progression_child = 3.0,
    progression_5_14 = 0.1,
    progression_15_64 = 0.25,
    progression_65_plus = 0.5,
    infectiousness_weights = (0.2, 0.5, 0.4, 1.0),
    pct_neg_tx_death = 0.4,
    disease_mortality_clin_lowinf = 0.025,
    disease_mortality_clin_inf = 0.4,
)

population0 = get_population(EQUILIBRIUM_YEAR)
u0 = seeded_initial_state(population0)

burnin_sol = simulate(
    params;
    tspan = (burnin_start, projection_start),
    u0 = u0,
    saveat = burnin_times,
    demography = :equilibrium,
)

projection_u0 = copy(burnin_sol.u[end])

equilibrium_sol = simulate(
    params;
    tspan = (projection_start, projection_end),
    u0 = copy(projection_u0),
    saveat = projection_times,
    demography = :equilibrium,
)

dynamic_sol = simulate(
    params;
    tspan = (projection_start, projection_end),
    u0 = copy(projection_u0),
    saveat = projection_times,
    demography = :dynamic,
)

equilibrium_summary = simulation_summary(equilibrium_sol)
dynamic_summary = simulation_summary(dynamic_sol)

equilibrium_demography = demographic_summary(equilibrium_summary)
dynamic_demography = demographic_summary(dynamic_summary)
```

Useful normalised comparisons include:

- TB incidence per 100,000;
- TB mortality per 100,000;
- relative differences in incidence or mortality between demographic scenarios;
- age-specific incidence rates;
- median population age;
- proportion of the population aged 65+;
- other age-structure summaries.

For example, the relative effect of dynamic demography on incidence can be calculated as:

```julia
relative_difference = 100 .* (
    dynamic_summary.incidence_per_100k ./
    equilibrium_summary.incidence_per_100k .- 1
)
```

A value of `5` corresponds to incidence being 5% higher under dynamic than equilibrium demography at that time point.

## Demography

The demographic component tracks the population by single year of age and can operate in two modes.

### Equilibrium demography

```julia
demography = :equilibrium
```

This mode provides a stable reference demographic regime and is used for long burn-in simulations and as the counterfactual comparison for the ageing analysis.

### Dynamic demography

```julia
demography = :dynamic
```

This mode applies time-varying age-specific demographic changes during the projection period. The implementation is designed to reproduce the population projections used by the project while updating the epidemiological state consistently as the population changes.

The demographic implementation has been checked against the underlying population projections before being used for epidemiological comparisons.

## Initial conditions and burn-in

The convenience function

```julia
seeded_initial_state(population)
```

constructs an initial epidemiological state from an age-specific population vector and seeds TB infection/disease into the population.

For scientific analyses, this initial state is normally not used directly as the projection starting point. Instead, the model is run through a long burn-in under equilibrium demography so that the epidemiological state approaches a model-consistent equilibrium.

The terminal state of the burn-in is then copied into each projection scenario.

## Parameters

Model parameters are constructed with:

```julia
make_parameters(CONTACT; ...)
```

The parameter set includes quantities governing transmission, age-specific progression, infectiousness, containment/clearance, clinical progression, recovery, detection/treatment, and TB-related mortality.

Several parameters can vary by broad age group. Current calibration work includes age-specific progression parameters for:

- children younger than 5 years;
- ages 5–14 years;
- ages 15–64 years;
- ages 65 years and older.

The contact matrix is supplied when parameters are constructed, allowing transmission to depend on age-specific mixing patterns.

## Calibration

The repository contains calibration scripts used to fit selected model parameters to age-specific TB burden estimates.

The current calibration is intentionally regarded as **provisional**. It is sufficient for model development and exploratory demographic comparisons but is not yet intended as a definitive epidemiological fit.

Calibration work to date has explored parameters including:

- transmission intensity (`beta`);
- age-specific progression rates;
- age-specific mixing/contact effects.

Optimisation has been performed using numerical optimisation in Julia, including Nelder–Mead searches. Age-specific estimated TB case data are used as calibration targets.

The external burden estimates used for calibration are reported with uncertainty intervals. A future refinement is to propagate this uncertainty more explicitly through the calibration likelihood rather than calibrating only to point estimates.

## Model outputs

The simulation summary utilities expose both aggregate and age-specific outputs.

### Epidemiological outputs

| Output               | Description                        |
| -------------------- | ---------------------------------- |
| `population`         | Total simulated population         |
| `population_by_age`  | Population by model age            |
| `incidence`          | Incident TB events                 |
| `incidence_by_age`   | Incident TB events by age          |
| `incidence_per_100k` | Population-normalised TB incidence |
| `deaths`             | TB-related deaths                  |
| `deaths_by_age`      | TB-related deaths by age           |
| `deaths_per_100k`    | Population-normalised TB mortality |

### Demographic outputs

| Output         | Description                                         |
| -------------- | --------------------------------------------------- |
| `population`   | Total population                                    |
| `median_age`   | Population median age                               |
| `prop_under_5` | Proportion of the population younger than 5 years   |
| `prop_65_plus` | Proportion of the population aged 65 years or older |

## Repository organisation

The repository follows a conventional Julia package layout.

```text
TBKiribatiJulia/
├── src/          # Package implementation
├── scripts/      # Calibration, validation, and scientific analyses
├── data/         # Model input data used by the package/project
├── test/         # Automated tests
├── Project.toml  # Julia project dependencies
└── README.md
```

The main package source is exposed through `src/TBKiribatiJulia.jl`, with implementation separated into model, parameter, demographic, simulation, and supporting utilities.

The `scripts/` directory contains research workflows that use the package rather than forming part of the core package API. In particular, `compare_demography.jl` contains the principal equilibrium-versus-dynamic-demography projection experiment.

## Recommended analysis outputs

For the demographic comparison, the primary outputs should remain population-normalised because the equilibrium scenario intentionally suppresses population growth.

Recommended figures include:

1. overall TB incidence per 100,000 under equilibrium and dynamic demography;
2. relative difference in incidence between scenarios;
3. TB mortality per 100,000;
4. age-specific TB incidence rates;
5. median population age;
6. proportion of the population aged 65+.

A selected-year summary table can combine epidemiological and demographic quantities for years such as 2025, 2050, 2075, and 2100.

## Testing

Run the package test suite from the project root with:

```julia
using Pkg
Pkg.test()
```

The tests cover core model and demographic functionality and should be run after changes to the simulation or demographic implementation.

## Development priorities

The immediate development priorities are:

- complete the scientific analysis of equilibrium versus dynamic demography;
- produce publication-quality plots and summary tables;
- refine calibration and uncertainty treatment;
- expand and update package documentation;
- continue consolidating reusable functionality into `src/` while keeping analysis-specific code in `scripts/`.

Potential future model extensions include further refinement of TB mortality, calibration, uncertainty propagation, and intervention scenarios.

## Reproducibility

The project is intended to be run from its Julia project environment. For reproducible analyses, activate the repository environment and instantiate dependencies before running scripts:

```bash
julia --project=.
```

```julia
using Pkg
Pkg.instantiate()
```

Scientific scripts should be run from the repository root unless otherwise documented.

## Citation

This package is under active development. A formal citation should be added when the associated methods and/or application manuscript is available.

## Author

Michael Meehan  
James Cook University

## Repository

https://github.com/michaeltmeehan/tb-kiribati-julia
