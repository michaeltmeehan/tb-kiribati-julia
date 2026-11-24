
using TBKiribatiJulia
using DataFrames, CSV
using Distributions: Binomial, logpdf

const REC_DIAG_DIR = joinpath(
    "results", "DoddExtended",
    "calibration_component_comparison"
)

mkpath(REC_DIAG_DIR)

# The comparison script must have been run already.
@assert length(fits) == 3
@assert length(problems) == 3

full_problem = problems[3]
observations = full_problem.recurrence.rows

recurrence_rows = NamedTuple[]
consistency_rows = NamedTuple[]

for (i, fit) in enumerate(fits)

    experiment = ("A", "B", "C")[i]

    output = TBKiribatiJulia.predict(
        full_problem,
        fit.parameters;
        variables=(
            :notifications,
            :new_notifications,
            :recurrent_notifications,
        ),
    )

    # Check notification accounting for every year.
    for (j, year) in enumerate(output.years)

        total = sum(output.notifications[:, j])
        new = sum(output.new_notifications[:, j])
        recurrent = sum(
            output.recurrent_notifications[:, j]
        )

        push!(consistency_rows, (
            experiment=experiment,
            year=year,
            total=total,
            new=new,
            recurrent=recurrent,
            accounting_error=total-new-recurrent,
        ))

        @assert isapprox(
            total, new + recurrent;
            atol=1e-5,
            rtol=1e-8,
        )
    end

    for obs in observations

        j = findfirst(==(obs.year), output.years)
        @assert !isnothing(j)

        model_total = sum(
            output.notifications[:, j]
        )

        model_recurrent = sum(
            output.recurrent_notifications[:, j]
        )

        @assert model_total > 0

        q_model = model_recurrent / model_total
        q_observed = obs.recurrent / obs.total

        # Exact likelihood used by recurrence.jl.
        ll = logpdf(
            Binomial(obs.total, q_model),
            obs.recurrent,
        )

        # Binomial saturated likelihood:
        # probability equals the observed fraction.
        ll_saturated = logpdf(
            Binomial(obs.total, q_observed),
            obs.recurrent,
        )

        push!(recurrence_rows, (
            experiment=experiment,
            year=obs.year,
            observed_total=obs.total,
            observed_recurrent=obs.recurrent,
            observed_fraction=q_observed,
            predicted_total=model_total,
            predicted_recurrent=model_recurrent,
            predicted_fraction=q_model,
            expected_recurrent_given_observed_total=
            obs.total*q_model,
            loglik=ll,
            deviance=2*(ll_saturated-ll),
        ))
    end
end

rec_diag = DataFrame(recurrence_rows)
counter_check = DataFrame(consistency_rows)

CSV.write(
    joinpath(REC_DIAG_DIR, "recurrence_diagnostic.csv"),
    rec_diag,
)

CSV.write(
    joinpath(REC_DIAG_DIR, "recurrence_counter_check.csv"),
    counter_check,
)

# --------------------------------------------------
# Summary across all observed recurrence years
# --------------------------------------------------

summary_rows = NamedTuple[]

for experiment in ("A", "B", "C")

    d = filter(
        r -> r.experiment == experiment,
        rec_diag,
    )

    push!(summary_rows, (
        experiment=experiment,
        observed_recurrent=sum(d.observed_recurrent),
        observed_total=sum(d.observed_total),
        observed_fraction=
        sum(d.observed_recurrent) /
            sum(d.observed_total),
        expected_recurrent=sum(
            d.expected_recurrent_given_observed_total
        ),
        predicted_fraction_weighted=
        sum(d.expected_recurrent_given_observed_total) /
            sum(d.observed_total),
        loglik=sum(d.loglik),
        total_deviance=sum(d.deviance),
    ))
end

rec_summary = DataFrame(summary_rows)

CSV.write(
    joinpath(REC_DIAG_DIR, "recurrence_summary.csv"),
    rec_summary,
)

println("\nRECURRENCE SUMMARY")
display(rec_summary)

println("\nRECURRENCE FRACTIONS BY YEAR")
display(select(
    rec_diag,
    :experiment,
    :year,
    :observed_fraction,
    :predicted_fraction,
    :observed_recurrent,
    :expected_recurrent_given_observed_total,
))

println("\nLARGEST RECURRENCE DEVIANCES")
display(first(
    sort(rec_diag, :deviance; rev=true),
    15,
))

println("\nNOTIFICATION COUNTER CONSISTENCY")
println(
    "Maximum absolute accounting error: ",
    maximum(abs.(counter_check.accounting_error))
)


using Distributions: Multinomial, logpdf

age_rows = NamedTuple[]

for experiment in ("A", "B", "C")
    d = filter(
        r -> r.experiment == experiment &&
            2013 <= r.year <= 2023,
        notification_rows,
    )

    for year in unique(d.year)
        yr = filter(r -> r.year == year, d)

        length(yr.observed) == 8 || continue

        observed = Int.(yr.observed)
        predicted = Float64.(yr.expected)

        proportions = predicted ./ sum(predicted)

        ll = logpdf(
            Multinomial(sum(observed), proportions),
            observed,
        )

        observed_proportions =
            observed ./ sum(observed)

        ll_saturated = logpdf(
            Multinomial(
                sum(observed),
                observed_proportions
            ),
            observed,
        )

        push!(age_rows, (
            experiment=experiment,
            year=year,
            composition_loglik=ll,
            composition_deviance=2*(ll_saturated-ll),
        ))
    end
end

age_comparison = DataFrame(age_rows)

display(combine(
    groupby(age_comparison, :experiment),
    :composition_deviance => sum =>
        :total_composition_deviance,
))

CSV.write(
    joinpath(
        REC_DIAG_DIR,
        "age_composition_diagnostic.csv",
    ),
    age_comparison,
)
