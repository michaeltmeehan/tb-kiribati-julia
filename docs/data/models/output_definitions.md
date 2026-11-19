# Epidemiological output definitions

## Direct event outputs

All event outputs are annual counts derived from cumulative event counters. For calendar year \(y\), the increment is measured from the post-demographic state at \(y\) to the pre-demographic state at \(y+1\).

- **`disease_onsets`**: New episodes of active TB disease, including fast progression, slow progression and relapse. Not equivalent to new M. tuberculosis infections.
- **`treatment_initiations`**: Entry into the treatment compartment following modelled detection.
- **`tb_deaths`**: Sum of untreated and treatment TB-death counters. Excludes background demographic mortality.
- **`treatment_deaths`**: Modelled TB deaths occurring during treatment; not necessarily equivalent to all-cause deaths in a reported treatment cohort.

## Epidemiological stocks

Stocks are evaluated at the post-demographic state at the start of each requested calendar year.

- **`untreated_active_prevalence`**: Individuals occupying untreated active TB compartments.
- **`latent_prevalence`**: Individuals occupying early or late latent infection compartments.
- **`treatment_prevalence`**: Individuals occupying the treatment compartment.
- **`active_prevalence`**: Compatibility alias for `untreated_active_prevalence`; does not include individuals receiving treatment.

These are population counts, not prevalence proportions.

## Existing observation-facing outputs

For **Dodd**, `incidence`, `notifications` and `mortality` extrapolate directly modelled pulmonary events to all-form TB using the fixed `pulmonary_fraction`.

For **DoddExtended**, all three disease presentations are modelled explicitly, so the corresponding outputs equal the direct event counts.

Both models currently identify notification with detection followed immediately by treatment initiation. No separate reporting probability or diagnosis-to-treatment delay is modelled.

## Treatment history and recurrence

DoddExtended classifies notifications by the model's treatment-history states. These categories are not automatically equivalent to WHO reported new and recurrent case definitions.

Relapse is a disease-onset event, whereas recurrent notification is a subsequent detection/treatment event. They should not be treated as interchangeable.

## Disease presentation

DoddExtended's smear-positive, smear-negative and extrapulmonary outputs correspond to its explicitly modelled presentation compartments.

The presentation probabilities used at disease onset are derived from historical notification data. If detection differs by presentation, these observed proportions need not equal underlying disease-onset proportions.

## Model and observation separation

Direct events and stocks are produced by the mechanistic model. Reporting probabilities, extrapolation to unmodelled disease presentations, surveillance classification and measurement error should ultimately be handled by an explicit observation layer during inference consolidation.

Existing output aliases are retained temporarily for backward compatibility.
