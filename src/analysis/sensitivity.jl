using LinearAlgebra: rank, qr
using Statistics: cor
using StatsBase: corspearman, tiedrank

"""
    sensitivity_data(result; year, parameters, outcomes)

Extract aligned parameter draws and national annual outcomes
from a simulation ensemble.

Failed simulations are excluded. No transformation or
normalisation of the values is performed.
"""
function sensitivity_data(
    result;
    year::Integer,
    parameters=keys(first(result.draws)),
    outcomes=(:incidence, :notifications, :mortality),
)
    isempty(result.draws) &&
        throw(ArgumentError("Ensemble is empty"))

    indices = findall(!isnothing, result.outputs)

    isempty(indices) &&
        throw(ArgumentError("No successful simulations"))

    parameter_names = Tuple(parameters)
    outcome_names = Tuple(outcomes)

    parameter_values = map(parameter_names) do name
        Float64[
            getproperty(result.draws[i], name)
            for i in indices
        ]
    end

    outcome_values = map(outcome_names) do name
        map(indices) do i
            out = result.outputs[i]
            j = findfirst(==(year), out.years)

            isnothing(j) &&
                throw(ArgumentError("Year $year not found"))

            Float64(sum(getproperty(out, name)[:, j]))
        end
    end

    params = NamedTuple{parameter_names}(parameter_values)
    outs = NamedTuple{outcome_names}(outcome_values)

    for values in (parameter_values..., outcome_values...)
        all(isfinite, values) ||
            throw(ArgumentError(
                "Nonfinite parameter or outcome values"
            ))
    end

    return (
        parameters=params,
        outcomes=outs,
        indices=indices,
        year=year,
    )
end


"""
    sensitivity_correlations(data)

Calculate the Spearman correlation between every parameter
and outcome, and between every pair of parameters.

Returns named axes and correlation matrices.
"""
function sensitivity_correlations(data)

    pnames = collect(keys(data.parameters))
    onames = collect(keys(data.outcomes))

    np = length(pnames)
    no = length(onames)

    parameter_correlations = Matrix{Float64}(undef, np, np)
    outcome_correlations = Matrix{Float64}(undef, no, np)

    # Constant variables have undefined correlations.
    function rankcor(x, y)
        if all(==(first(x)), x) ||
            all(==(first(y)), y)
            return NaN
        end
        return corspearman(x, y)
    end

    for i in 1:np, j in 1:np
        parameter_correlations[i, j] = rankcor(
            data.parameters[pnames[i]],
            data.parameters[pnames[j]],
        )
    end

    for i in 1:no, j in 1:np
        outcome_correlations[i, j] = rankcor(
            data.outcomes[onames[i]],
            data.parameters[pnames[j]],
        )
    end

    return (
        parameter_names=pnames,
        outcome_names=onames,
        parameter_correlations=parameter_correlations,
        outcome_correlations=outcome_correlations,
    )
end


"""
    sensitivity_prcc(data)

Compute partial rank correlations between each parameter
and each outcome, adjusting for all other parameters.

Returns a matrix with rows = outcomes and columns = parameters.
"""
function sensitivity_prcc(data)

    pnames = collect(keys(data.parameters))
    onames = collect(keys(data.outcomes))

    np = length(pnames)
    no = length(onames)

    np > 0 || throw(ArgumentError("No parameters"))
    no > 0 || throw(ArgumentError("No outcomes"))

    n = length(first(values(data.parameters)))

    n > np + 1 ||
        throw(ArgumentError("Insufficient samples"))

    # Parameter rank matrix
    X = Matrix{Float64}(undef, n, np)

    for (j, name) in enumerate(pnames)
        x = data.parameters[name]

        length(x) == n ||
            throw(DimensionMismatch("Parameter lengths differ"))

        all(isfinite, x) ||
            throw(ArgumentError("Nonfinite parameter: $name"))

        length(unique(x)) > 1 ||
            throw(ArgumentError("Constant parameter: $name"))

        X[:, j] = tiedrank(x)
    end

    # Outcome rank matrix
    Y = Matrix{Float64}(undef, n, no)

    for (i, name) in enumerate(onames)
        y = data.outcomes[name]

        length(y) == n ||
            throw(DimensionMismatch("Outcome lengths differ"))

        all(isfinite, y) ||
            throw(ArgumentError("Nonfinite outcome: $name"))

        length(unique(y)) > 1 ||
            throw(ArgumentError("Constant outcome: $name"))

        Y[:, i] = tiedrank(y)
    end

    coefficients = Matrix{Float64}(undef, no, np)

    for j in 1:np

        other = [k for k in 1:np if k != j]

        Z = hcat(ones(n), X[:, other])

        rank(Z) == size(Z, 2) ||
            throw(ArgumentError(
                "Rank-deficient adjustment matrix"
            ))

        F = qr(Z)

        x = X[:, j]
        rx = x - Z * (F \ x)

        for i in 1:no
            y = Y[:, i]
            ry = y - Z * (F \ y)

            coefficients[i, j] = cor(rx, ry)
        end
    end

    return (
        parameter_names=pnames,
        outcome_names=onames,
        prcc=coefficients,
        n_samples=n,
    )
end