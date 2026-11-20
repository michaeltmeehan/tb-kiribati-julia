
using Distributions: logpdf, Normal, truncated
using SciMLBase: successful_retcode

"""
    CalibrationProblem(
        model,
        demography,
        observations;
        start_year,
        end_year,
        age_breaks,
        observation_model,
    )

Generic calibration configuration.

Model-specific simulation, parameter mapping and
output extraction are provided by model dispatch.
"""
struct CalibrationProblem{M,D,O,Q,P,S}
    model::M
    demography::D
    observations::O
    observation_model::Q
    prior::P
    simulation::S

    age_breaks::Vector{Int}
    start_year::Int
    end_year::Int

    treatment_deaths::Union{
        Nothing,TreatmentDeathObservation
    }

    recurrence::Union{
        Nothing,RecurrentTBObservation
    }

    site::Union{Nothing,SiteTBObservation}
end


function CalibrationProblem(
    model::AbstractTBModel,
    demography::AbstractDemography,
    observations;
    start_year::Integer,
    end_year::Integer,
    treatment_deaths=nothing,
    recurrence=nothing,
    site=nothing,
    age_breaks,
    observation_model=NB2Observation(
        truncated(Normal(0, 0.3), 0, Inf)
    ),
    simulation_kwargs...,
)
    start_year < end_year ||
        throw(ArgumentError("Invalid simulation interval"))

    obs_data = collect(observations)

    isempty(obs_data) &&
        throw(ArgumentError("No observations supplied"))

    breaks = Int.(collect(age_breaks))
    layout = state_layout(model)

    length(breaks) >= 2 &&
        first(breaks) == 0 &&
        last(breaks) == layout.nages &&
        all(diff(breaks) .> 0) ||
        throw(ArgumentError(
            "Age breaks must strictly increase " *
                "from 0 to the final model age"
        ))

    ngroups = length(breaks) - 1
    years = [obs.year for obs in obs_data]

    length(unique(years)) == length(years) ||
        throw(ArgumentError(
            "Duplicate observation years"
        ))

    all(y -> start_year <= y < end_year, years) ||
        throw(ArgumentError(
            "Observation years outside simulation interval"
        ))

    for obs in obs_data
        n = length(obs.counts)

        n > 0 &&
            n == length(obs.groups) &&
            n == length(obs.labels) ||
            throw(DimensionMismatch(
                "Invalid observation dimensions"
            ))

        all(y -> y isa Integer && y >= 0, obs.counts) ||
            throw(ArgumentError(
                "Invalid notification counts"
            ))

        used = Set{Int}()

        for group in obs.groups
            isempty(group) &&
                throw(ArgumentError("Empty age group"))

            for g in group
                g isa Integer && 1 <= g <= ngroups ||
                    throw(ArgumentError(
                        "Invalid age group index"
                    ))

                g in used &&
                    throw(ArgumentError(
                        "Overlapping observed age groups " *
                            "in year $(obs.year)"
                    ))

                push!(used, g)
            end
        end
    end

    if !isnothing(recurrence)
        all(
            r -> start_year <= r.year < end_year,
            recurrence.rows,
        ) || throw(ArgumentError(
            "Recurrence year outside simulation interval"
        ))
    end

    if !isnothing(site)
        all(r -> start_year <= r.year < end_year, site.rows) ||
            throw(ArgumentError("Site year outside simulation interval"))
    end

    prior = merge(
        priors(model),
        observation_priors(observation_model),
    )

    simulation = prepare_simulation(
        model,
        demography;
        start_year,
        end_year,
        simulation_kwargs...,
    )

    return CalibrationProblem(
        model,
        demography,
        obs_data,
        observation_model,
        prior,
        simulation,
        breaks,
        Int(start_year),
        Int(end_year),
        treatment_deaths,
        recurrence,
        site,
    )
end


"""
    model_draw(problem, theta)

Extract epidemiological model parameters from the
full calibration parameter set.
"""
function model_draw(
    problem::CalibrationProblem,
    theta::NamedTuple,
)
    names = keys(priors(problem.model))

    all(name -> haskey(theta, name), names) ||
        throw(ArgumentError(
            "Missing epidemiological parameters"
        ))

    return NamedTuple{names}(
        Tuple(
            getproperty(theta, name)
            for name in names
        )
    )
end


calibration_diagnostic_variables(model::AbstractTBModel) =
    default_output_variables(model)

calibration_diagnostic_variables(model::DoddExtendedTBModel) = (
    default_output_variables(model)...,
    :active_prevalence,
    :untreated_active_prevalence,
    :latent_prevalence,
    :treatment_prevalence,
    :disease_onsets,
    :treatment_initiations,
    :tb_deaths,
    :treatment_deaths,
)

"""
    predict(problem, theta; variables=:likelihood, years=nothing)

Simulate once and extract age-aggregated annual predictions.

`variables=:likelihood` extracts only likelihood-required outputs.
`variables=nothing` extracts the full diagnostic output set.
An explicit tuple selects additional outputs.

`years` can extend the prediction period, provided all observation
years remain included and the simulation interval is respected.
"""
function predict(
    problem::CalibrationProblem,
    theta::NamedTuple;
    variables=:likelihood,
    years=nothing,
)
    required = Symbol[:notifications]
    observation_years = [obs.year for obs in problem.observations]

    if !isnothing(problem.recurrence)
        push!(required, :recurrent_notifications)
        append!(
            observation_years,
            (r.year for r in problem.recurrence.rows),
        )
    end

    if !isnothing(problem.site)
        append!(
            required,
            (:pulmonary_notifications, :extrapulmonary_notifications),
        )
        append!(
            observation_years,
            (r.year for r in problem.site.rows),
        )
    end

    sort!(observation_years)
    unique!(observation_years)

    requested = if variables === :likelihood
        Tuple(required)
    elseif isnothing(variables)
        calibration_diagnostic_variables(problem.model)
    elseif variables isa Tuple && all(v -> v isa Symbol, variables)
        variables
    else
        throw(ArgumentError(
            "variables must be :likelihood, nothing, or a tuple of Symbols"
        ))
    end

    selected = Tuple(unique(vcat(collect(requested), required)))

    prediction_years = isnothing(years) ?
                       observation_years : Int.(collect(years))

    isempty(prediction_years) &&
        throw(ArgumentError("No prediction years"))

    length(unique(prediction_years)) == length(prediction_years) ||
        throw(ArgumentError("Duplicate prediction years"))

    all(y -> problem.start_year <= y < problem.end_year, prediction_years) ||
        throw(ArgumentError("Prediction year outside simulation interval"))

    all(y -> y in prediction_years, observation_years) ||
        throw(ArgumentError(
            "Prediction years must include all observation years"
        ))

    sort!(prediction_years)

    sol = simulate_from_draw(
        problem.simulation,
        model_draw(problem, theta),
    )

    successful_retcode(sol) ||
        throw(ErrorException("ODE simulation failed: $(sol.retcode)"))

    return model_outputs(
        problem.model,
        sol;
        variables=selected,
        years=prediction_years,
        age_breaks=problem.age_breaks,
    )
end


# Preserve the existing public convenience method.
notification_fit_rows(
    problem::CalibrationProblem,
    output,
) = notification_fit_rows(
    problem.observations,
    problem.age_breaks,
    output,
)


"""
    calibration_loglikelihood(
        problem,
        theta;
        output=nothing,
    )

Evaluate the combined observation likelihood.

Supply previously computed model outputs to avoid
repeating the ODE simulation.
"""
function calibration_loglikelihood(
    problem::CalibrationProblem,
    theta::NamedTuple;
    output=nothing,
)
    out = isnothing(output) ?
          predict(problem, theta) :
          output

    ll_notifications = notification_loglikelihood(
        problem.observation_model,
        problem.observations,
        out,
        problem.age_breaks,
        theta,
    )

    isfinite(ll_notifications) || return -Inf

    ll_deaths = treatment_death_loglikelihood(
        problem.treatment_deaths,
        theta,
    )

    isfinite(ll_deaths) || return -Inf

    ll_recurrence = recurrence_loglikelihood(
        problem.recurrence,
        out,
    )

    isfinite(ll_recurrence) || return -Inf

    ll_site = site_loglikelihood(problem.site, out)

    isfinite(ll_site) || return -Inf

    return ll_notifications + ll_deaths + ll_recurrence + ll_site
end


"""
    calibration_logprior(problem, theta)

Prior density in original parameter coordinates.
"""
function calibration_logprior(
    problem::CalibrationProblem,
    theta::NamedTuple,
)
    all(
        name -> haskey(theta, name),
        keys(problem.prior),
    ) || throw(ArgumentError(
        "Missing calibration parameters"
    ))

    return sum(
        logpdf(dist, getproperty(theta, name))
        for (name, dist) in pairs(problem.prior)
    )
end


"""
    calibration_logposterior(problem, theta)

Unnormalised log posterior.
"""
function calibration_logposterior(
    problem::CalibrationProblem,
    theta::NamedTuple,
)
    lp = calibration_logprior(problem, theta)

    isfinite(lp) || return -Inf

    return lp + calibration_loglikelihood(
        problem,
        theta,
    )
end
