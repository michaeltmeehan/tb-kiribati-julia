
"""
    notification_fit_rows(observations, age_breaks, output)

Aggregate predicted notifications into the observed
age categories, including combined categories.

Returns one record per observed count.
"""
function notification_fit_rows(
    observations,
    age_breaks,
    output,
)
    hasproperty(output, :notifications) ||
        throw(ArgumentError(
            "Missing notification output"
        ))

    predicted = output.notifications

    size(predicted, 1) == length(age_breaks) - 1 ||
        throw(DimensionMismatch(
            "Age dimension mismatch"
        ))

    rows = NamedTuple[]

    for obs in observations
        j = findfirst(==(obs.year), output.years)

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
                expected=mu,
            ))
        end
    end

    return rows
end


"""
    notification_loglikelihood(
        observation_model,
        observations,
        output,
        age_breaks,
        theta,
    )

Evaluate the notification observation likelihood.

For annual random-effect models, observations from
the same year are evaluated jointly, with the
shared random effect integrated out.
"""
function notification_loglikelihood(
    observation_model,
    observations,
    output,
    age_breaks,
    theta,
)
    rows = notification_fit_rows(
        observations,
        age_breaks,
        output,
    )

    ll_notifications = 0.0
    offset = 1

    for observation in observations
        n = length(observation.counts)

        yearly_rows = @view rows[
            offset:(offset+n-1)
        ]

        term = observation_year_loglikelihood(
            observation_model,
            yearly_rows,
            theta,
        )

        isfinite(term) || return -Inf

        ll_notifications += term
        offset += n
    end

    return ll_notifications
end
