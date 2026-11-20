
using Statistics: mean, std

"""
    calibration_diagnostics(fit)

Return fitted predictions, detailed observation records,
annual totals across reported age groups, and Pearson
residuals.

Annual totals include only the reported age groups.
Pearson residuals use the marginal observation variance.
"""
function calibration_diagnostics(
    fit::AbstractCalibrationFit,
)

    output = predict(fit)

    rows = notification_fit_rows(
        fit.problem,
        output,
    )

    detailed = map(rows) do row

        variance = observation_variance(
            fit.problem.observation_model,
            row.expected,
            fit.parameters,
        )

        residual = row.observed - row.expected

        pearson = variance > 0 ?
                  residual / sqrt(variance) :
                  NaN

        return (
            row...,
            residual=residual,
            pearson_residual=pearson,
        )
    end

    years = [
        obs.year
        for obs in fit.problem.observations
    ]

    national = map(years) do year

        group = filter(
            row -> row.year == year,
            rows,
        )

        return (
            year=year,
            observed=sum(
                row.observed for row in group
            ),
            expected=sum(
                row.expected for row in group
            ),
        )
    end

    return (
        predictions=output,
        age=detailed,
        national=national,
        n_observations=length(rows),
        mean_pearson=mean(
            row.pearson_residual for row in detailed
        ),
        sd_pearson=std(
            row.pearson_residual for row in detailed
        ),
    )
end
