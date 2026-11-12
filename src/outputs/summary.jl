
using Statistics: quantile

"""
    summarise_ensemble(
        result;
        variables=(:incidence, :notifications, :mortality),
        probabilities=[0.025, 0.25, 0.5, 0.75, 0.975],
    )

Compute pointwise quantiles across successful ensemble simulations.

For each variable, returns an array with dimensions:

    age group × year × quantile

Failed simulations are excluded from quantile calculations
but included in the reported simulation counts.

All successful outputs must share the same years, age groups,
and array dimensions.
"""
function summarise_ensemble(
    result;
    variables::Tuple=(
        :incidence,
        :notifications,
        :mortality,
    ),
    probabilities::AbstractVector{<:Real}=[
        0.025, 0.25, 0.5, 0.75, 0.975
    ],
)
    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    probs = Float64.(probabilities)

    isempty(probs) &&
        throw(ArgumentError("No quantile probabilities supplied"))

    all(p -> isfinite(p) && 0 <= p <= 1, probs) ||
        throw(ArgumentError(
            "Quantile probabilities must lie in [0, 1]"
        ))

    issorted(probs) ||
        throw(ArgumentError(
            "Quantile probabilities must be sorted"
        ))

    isempty(variables) &&
        throw(ArgumentError("No output variables requested"))

    all(v -> v isa Symbol, variables) ||
        throw(ArgumentError("Output variables must be Symbols"))

    length(unique(variables)) == length(variables) ||
        throw(ArgumentError("Output variables must be unique"))

    n_draws = length(result.draws)

    length(result.outputs) == n_draws ||
        throw(DimensionMismatch(
            "Number of outputs does not match number of draws"
        ))

    # --------------------------------------------------------
    # Identify successful simulations
    # --------------------------------------------------------

    successful = [
        out for out in result.outputs
                if !isnothing(out)
    ]

    n_success = length(successful)
    n_failed = n_draws - n_success

    n_success > 0 ||
        throw(ArgumentError(
            "Cannot summarise an ensemble with no successful simulations"
        ))

    reference = first(successful)

    years = collect(reference.years)
    age_breaks = collect(reference.age_breaks)

    # --------------------------------------------------------
    # Validate output consistency
    # --------------------------------------------------------

    for out in successful

        out.years == years ||
            throw(ArgumentError(
                "Inconsistent output years across simulations"
            ))

        out.age_breaks == age_breaks ||
            throw(ArgumentError(
                "Inconsistent age groups across simulations"
            ))

    end


    # --------------------------------------------------------
    # Compute age-specific and national quantiles
    # --------------------------------------------------------

    summaries = map(variables) do variable

        hasproperty(reference, variable) ||
            throw(ArgumentError(
                "Output variable $variable does not exist"
            ))

        reference_values = getproperty(reference, variable)

        reference_values isa AbstractMatrix ||
            throw(ArgumentError(
                "$variable must be an age × year matrix"
            ))

        nages, nyears = size(reference_values)

        nyears == length(years) ||
            throw(DimensionMismatch(
                "$variable has inconsistent year dimensions"
            ))

        nages == length(age_breaks) - 1 ||
            throw(DimensionMismatch(
                "$variable has inconsistent age dimensions"
            ))

        for out in successful

            hasproperty(out, variable) ||
                throw(ArgumentError(
                    "Missing $variable in an ensemble output"
                ))

            values = getproperty(out, variable)

            size(values) == (nages, nyears) ||
                throw(DimensionMismatch(
                    "Inconsistent dimensions for $variable"
                ))

            all(isfinite, values) ||
                throw(ArgumentError(
                    "Nonfinite values found in $variable"
                ))
        end

        # Age × year × quantile
        age_summary = Array{Float64}(
            undef, nages, nyears, length(probs)
        )

        # Year × quantile
        total_summary = Matrix{Float64}(
            undef, nyears, length(probs)
        )

        samples = Vector{Float64}(undef, n_success)
        totals = Vector{Float64}(undef, n_success)

        for j in 1:nyears

            # National totals: sum ages within each simulation
            # BEFORE computing ensemble quantiles.
            for (k, out) in enumerate(successful)
                values = getproperty(out, variable)
                totals[k] = sum(@view values[:, j])
            end

            total_summary[j, :] .= quantile(totals, probs)

            # Age-specific quantiles
            for a in 1:nages

                for (k, out) in enumerate(successful)
                    samples[k] =
                        getproperty(out, variable)[a, j]
                end

                age_summary[a, j, :] .= quantile(samples, probs)
            end
        end

        return (
            ages=age_summary,
            totals=total_summary,
        )
    end

    output_summary = NamedTuple{variables}(
        map(x -> x.ages, summaries)
    )

    total_summary = NamedTuple{variables}(
        map(x -> x.totals, summaries)
    )

    return (
        years=years,
        age_breaks=age_breaks,
        probabilities=probs,
        n_draws=n_draws,
        n_success=n_success,
        n_failed=n_failed,
        failure_rate=n_failed / n_draws,
        outputs=output_summary,
        totals=total_summary,
    )
end
