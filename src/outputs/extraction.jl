
mutable struct OutputExtraction{S,L}
    solution::S
    layout::L
    windows::AnnualWindows
    counter_cache::Dict{Int,Any}
end

function OutputExtraction(
    solution,
    model::AbstractTBModel;
    years,
)
    return OutputExtraction(
        solution,
        state_layout(model),
        annual_windows(solution; years),
        Dict{Int,Any}(),
    )
end

function counter_output(
    extraction::OutputExtraction,
    counter::Integer,
)
    return get!(extraction.counter_cache, Int(counter)) do
        annual_events(
            extraction.solution,
            extraction.layout,
            counter,
            extraction.windows,
        )
    end
end


"""
Extract the sum of specified living compartments at the
post-demographic state of each requested calendar year.

Results have dimensions age × year.
"""
function stock_output(
    extraction::OutputExtraction,
    compartments,
)
    layout = extraction.layout
    sol = extraction.solution

    all(c -> 1 <= c <= layout.nepi, compartments) ||
        throw(ArgumentError(
            "Stock output requires living compartments"
        ))

    T = eltype(first(sol.u))
    years = extraction.windows.years

    result = zeros(T, layout.nages, length(years))

    for (j, i) in enumerate(extraction.windows.start_indices)
        u = sol.u[i]

        for a in 1:layout.nages
            for c in compartments
                result[a, j] +=
                    u[state_index(layout, a, c)]
            end
        end
    end

    return result
end


"""
    model_output(model, solution, variable; years,
                 age_breaks=nothing)

Extract one epidemiological output.

Model-specific methods dispatch on Val(variable).
"""
function model_output(
    model::AbstractTBModel,
    solution,
    variable::Symbol;
    years,
    age_breaks=nothing,
)
    extraction = OutputExtraction(
        solution, model; years
    )

    values = model_output(
        model, extraction, Val(variable)
    )

    return isnothing(age_breaks) ?
           values :
           aggregate_age_groups(values, age_breaks)
end


"""
    model_outputs(model, solution; variables, years,
                  age_breaks=nothing)

Extract only requested outputs.

Returns a NamedTuple containing metadata and one matrix
for each requested variable.
"""
function model_outputs(
    model::AbstractTBModel,
    solution;
    variables::Tuple=default_output_variables(model),
    years,
    age_breaks=nothing,
)
    isempty(variables) &&
        throw(ArgumentError("No outputs requested"))

    all(v -> v isa Symbol, variables) ||
        throw(ArgumentError("Output names must be Symbols"))

    length(unique(variables)) == length(variables) ||
        throw(ArgumentError("Duplicate output names"))

    extraction = OutputExtraction(
        solution, model; years
    )

    group(values) = isnothing(age_breaks) ?
                    values :
                    aggregate_age_groups(values, age_breaks)

    outputs = NamedTuple{
        variables
    }(
        map(variables) do variable
            group(model_output(
                model,
                extraction,
                Val(variable),
            ))
        end
    )

    return merge(
        (
            years=extraction.windows.years,
            age_breaks=isnothing(age_breaks) ?
                       collect(0:extraction.layout.nages) :
                       collect(age_breaks),
        ),
        outputs,
    )
end


function model_output(
    model::AbstractTBModel,
    extraction::OutputExtraction,
    ::Val{V},
) where {V}
    throw(ArgumentError(
        "Output $V is not defined for $(typeof(model))"
    ))
end
