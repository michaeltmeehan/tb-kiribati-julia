
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
end


function CalibrationProblem(
    model::AbstractTBModel,
    demography::AbstractDemography,
    observations;
    start_year::Integer,
    end_year::Integer,
    treatment_deaths=nothing,
    recurrence=nothing,
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


"""
    predict(problem, theta)

Simulate the model and calculate annual predictions
required by the calibration likelihood.
"""
function predict(
    problem::CalibrationProblem,
    theta::NamedTuple,
)
    sol = simulate_from_draw(
        problem.simulation,
        model_draw(problem, theta),
    )

    successful_retcode(sol) ||
        throw(ErrorException(
            "ODE simulation failed: $(sol.retcode)"
        ))

    years = [obs.year for obs in problem.observations]
    variables = (:notifications,)

    if !isnothing(problem.recurrence)
        append!(
            years,
            (r.year for r in problem.recurrence.rows),
        )

        variables = (
            :notifications,
            :recurrent_notifications,
        )
    end

    sort!(years)
    unique!(years)

    return model_outputs(
        problem.model,
        sol;
        variables,
        years,
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

    return ll_notifications + ll_deaths + ll_recurrence
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
