using CSV
using DataFrames
using CairoMakie: save

"""Write fitted parameters, observed/predicted tables and calibration figures."""
function save_calibration(fit::MAPFit, directory::AbstractString;
    diagnostics=calibration_diagnostics(fit),
    figures=(national=plot_calibration_national(fit; diagnostics),
        age=plot_calibration_by_age(fit; diagnostics)))
    mkpath(directory)
    parameters = DataFrame(parameter=string.(collect(keys(fit.parameters))),
        estimate=Float64.(collect(values(fit.parameters))))
    CSV.write(joinpath(directory, "map_parameters.csv"), parameters)
    CSV.write(joinpath(directory, "map_notification_fit.csv"), DataFrame(diagnostics.age))
    CSV.write(joinpath(directory, "map_national_fit.csv"), DataFrame(diagnostics.national))
    metadata = DataFrame(metric=["log_prior", "log_likelihood", "log_posterior",
        "converged", "n_observations", "mean_pearson", "sd_pearson"],
        value=string.([fit.log_prior, fit.log_likelihood, fit.log_posterior,
            fit.converged, diagnostics.n_observations,
            diagnostics.mean_pearson, diagnostics.sd_pearson]))
    CSV.write(joinpath(directory, "map_summary.csv"), metadata)
    for (name, fig) in pairs(figures)
        save(joinpath(directory, "map_$(name).png"), fig)
    end
    return directory
end
