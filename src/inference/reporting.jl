
using CSV
using DataFrames
using CairoMakie: save

function save_calibration(
    fit::AbstractCalibrationFit,
    directory::AbstractString;
    diagnostics=calibration_diagnostics(fit),
    figures=(
        national=plot_calibration_national(
            fit; diagnostics=diagnostics
        ),
        age=plot_calibration_by_age(
            fit; diagnostics=diagnostics
        ),
    ),
)
    mkpath(directory)

    prefix = fit isa MAPFit ? "map" : "mle"

    parameter_table = DataFrame(
        parameter=string.(collect(keys(fit.parameters))),
        estimate=Float64.(collect(Base.values(fit.parameters))),
    )

    CSV.write(
        joinpath(directory, "$(prefix)_parameters.csv"),
        parameter_table,
    )

    CSV.write(
        joinpath(directory, "$(prefix)_notification_fit.csv"),
        DataFrame(diagnostics.age),
    )

    CSV.write(
        joinpath(directory, "$(prefix)_national_fit.csv"),
        DataFrame(diagnostics.national),
    )

    metrics = [
        "log_likelihood",
        "converged",
        "n_observations",
        "mean_pearson",
        "sd_pearson",
    ]

    values = Any[
        fit.log_likelihood,
        fit.converged,
        diagnostics.n_observations,
        diagnostics.mean_pearson,
        diagnostics.sd_pearson,
    ]

    if fit isa MAPFit
        append!(metrics, [
            "log_prior",
            "log_posterior",
        ])
        append!(values, [
            fit.log_prior,
            fit.log_posterior,
        ])
    else
        push!(metrics, "n_parameters")
        push!(values, fit.n_parameters)
    end

    CSV.write(
        joinpath(directory, "$(prefix)_summary.csv"),
        DataFrame(
            metric=metrics,
            value=string.(values),
        ),
    )

    for (name, fig) in pairs(figures)
        save(
            joinpath(
                directory,
                "$(prefix)_$(name).png",
            ),
            fig,
        )
    end

    return directory
end
