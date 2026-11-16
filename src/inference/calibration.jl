
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

Model-specific simulation, parameter mapping and output
extraction are provided by existing model dispatch.
"""
struct CalibrationProblem{M,D,O,A,Q,P,T,R}
    model::M
    demography::D
    observations::O
    age_breaks::A
    observation_model::Q
    prior::P
    start_year::Int
    end_year::Int

    treatment_deaths::T
    recurrence::R
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
)
    start_year < end_year ||
        throw(ArgumentError(
            "Invalid simulation interval"
        ))

    isempty(observations) &&
        throw(ArgumentError(
            "No observations supplied"
        ))

    years = [obs.year for obs in observations]

    length(unique(years)) == length(years) ||
        throw(ArgumentError(
            "Repeated observation year"
        ))

    all(
        y -> start_year <= y && y + 1 <= end_year,
        years,
    ) || throw(ArgumentError(
        "Observation year outside simulation interval"
    ))

    ngroups = length(age_breaks) - 1

    for obs in observations

        if length(obs.counts) != length(obs.groups) ||
            length(obs.counts) != length(obs.labels)
            throw(DimensionMismatch(
                "Observation dimensions differ"
            ))
        end

        for group in obs.groups
            isempty(group) &&
                throw(ArgumentError(
                    "Empty observation age group"
                ))

            all(g -> 1 <= g <= ngroups, group) ||
                throw(ArgumentError(
                    "Invalid age group index"
                ))
        end
    end

    prior = merge(
        priors(model),
        observation_priors(observation_model),
    )

    return CalibrationProblem(
        model,
        demography,
        observations,
        collect(age_breaks),
        observation_model,
        prior,
        Int(start_year),
        Int(end_year),
        treatment_deaths,
        recurrence,
    )
end


"""
    model_draw(problem, theta)

Extract only epidemiological model parameters from
the full calibration parameter set.
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
in the standard calibration age groups.
"""
function predict(
    problem::CalibrationProblem,
    theta::NamedTuple,
)
    sol = simulate_from_draw(
        problem.model,
        model_draw(problem, theta),
        problem.demography;
        start_year=problem.start_year,
        end_year=problem.end_year,
    )

    successful_retcode(sol.retcode) ||
        throw(ErrorException(
            "ODE simulation failed: $(sol.retcode)"
        ))


    years = [obs.year for obs in problem.observations]

    if !isnothing(problem.recurrence)
        append!(
            years,
            [r.year for r in problem.recurrence.rows],
        )
    end

    sort!(unique!(years))

    return model_outputs(
        problem.model,
        sol;
        years=years,
        age_breaks=problem.age_breaks,
    )

end


"""
    notification_fit_rows(problem, output)

Aggregate model predictions into the observed
age categories, including combined categories.

Returns one record per observed count.
"""
function notification_fit_rows(
    problem::CalibrationProblem,
    output,
)
    hasproperty(output, :notifications) ||
        throw(ArgumentError(
            "Missing notification output"
        ))

    predicted = output.notifications

    size(predicted, 1) ==
        length(problem.age_breaks) - 1 ||
        throw(DimensionMismatch(
            "Age dimension mismatch"
        ))

    rows = NamedTuple[]

    for obs in problem.observations

        j = findfirst(
            ==(obs.year),
            output.years,
        )

        isnothing(j) &&
            throw(ArgumentError(
                "Missing prediction year $(obs.year)"
            ))

        for (label, y, groups) in zip(
            obs.labels,
            obs.counts,
            obs.groups,
        )
            mu = sum(
                predicted[g, j]
                for g in groups
            )

            push!(rows, (
                year=obs.year,
                age_group=label,
                observed=y,
                expected=Float64(mu),
            ))
        end
    end

    return rows
end


"""
    calibration_loglikelihood(
        problem,
        theta;
        output=nothing,
    )

Evaluate the observation likelihood.

Supply a previously computed model output to avoid
repeating the ODE simulation.

Annual random effects are integrated jointly across
all reported age groups within each year.
"""
function calibration_loglikelihood(
    problem::CalibrationProblem,
    theta::NamedTuple;
    output=nothing,
)
    out = isnothing(output) ?
          predict(problem, theta) :
          output

    rows = notification_fit_rows(problem, out)

    ll = 0.0
    offset = 1

    for observation in problem.observations

        n = length(observation.counts)

        yearly_rows = @view rows[
            offset:(offset+n-1)
        ]

        term = observation_year_loglikelihood(
            problem.observation_model,
            yearly_rows,
            theta,
        )

        isfinite(term) || return -Inf

        ll += term
        offset += n
    end


    # Conditional treatment-death likelihood.
    ll_deaths = treatment_death_loglikelihood(
        problem.treatment_deaths,
        theta,
    )

    isfinite(ll_deaths) || return -Inf

    # Conditional recurrent-notification likelihood.
    ll_recurrence = recurrence_loglikelihood(
        problem.recurrence,
        out,
    )

    isfinite(ll_recurrence) || return -Inf

    return ll + ll_deaths + ll_recurrence

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
