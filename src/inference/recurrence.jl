
using Distributions: Binomial, logpdf

"""
National recurrent notifications conditional on the
observed total notifications in each year.

The two input datasets are the cleaned WHO national
and recurrent notification CSVs.
"""
struct RecurrentTBObservation{R}
    rows::R
end

function RecurrentTBObservation(
    recurrent_notifications,
    national_notifications,
)
    totals = Dict(
        r.year => r.cases for r in national_notifications
    )

    rows = NamedTuple[]

    for r in recurrent_notifications
        year = r.year

        haskey(totals, year) ||
            throw(ArgumentError(
                "Missing national notifications for $year"
            ))

        total = totals[year]
        recurrent = r.cases

        (
            total isa Integer &&
                recurrent isa Integer &&
                0 <= recurrent <= total
        ) || throw(ArgumentError(
            "Invalid recurrent or national count in $year"
        ))

        push!(rows, (
            year=year,
            total=total,
            recurrent=recurrent,
        ))
    end

    length(unique(r.year for r in rows)) ==
        length(rows) ||
        throw(ArgumentError("Duplicate recurrence year"))

    sort!(rows; by=r -> r.year)

    return RecurrentTBObservation(rows)
end


recurrence_loglikelihood(::Nothing, output) = 0.0

"""
Evaluate the conditional recurrence likelihood.

Requires output.recurrent_notifications, a matrix
with exactly the same age/year dimensions as
output.notifications.

These recurrent predictions are NOT currently
produced by the unaugmented Dodd model.
"""
function recurrence_loglikelihood(
    obs::RecurrentTBObservation,
    output,
)
    hasproperty(output, :recurrent_notifications) ||
        throw(ArgumentError(
            "Model must provide recurrent_notifications"
        ))

    total_pred = output.notifications
    recurrent_pred = output.recurrent_notifications

    size(total_pred) == size(recurrent_pred) ||
        throw(DimensionMismatch(
            "Total and recurrent prediction shapes differ"
        ))

    ll = 0.0

    for r in obs.rows
        j = findfirst(==(r.year), output.years)

        isnothing(j) &&
            throw(ArgumentError(
                "Missing prediction year $(r.year)"
            ))

        total = sum(@view total_pred[:, j])
        recurrent = sum(@view recurrent_pred[:, j])

        if !(isfinite(total) && total > 0 &&
            isfinite(recurrent) &&
            0 <= recurrent <= total)
            return -Inf
        end

        p = recurrent / total

        ll += logpdf(
            Binomial(r.total, p),
            r.recurrent,
        )
    end

    return ll
end
