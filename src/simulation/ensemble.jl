
using SciMLBase: successful_retcode

"""
    simulate_ensemble(setup, draws;
                      output_years,
                      variables=default_output_variables(setup.model),
                      age_breaks=nothing)

Evaluate parameter draws using a reusable PreparedSimulation.

Every draw is independently processed through:
    make_parameters -> initial_state -> simulate -> model_outputs

Failed draws retain their original positions in the output
collection and are recorded without resampling.

Only requested output variables are calculated.
"""
function simulate_ensemble(
    setup::PreparedSimulation,
    draws;
    output_years,
    variables::Tuple=default_output_variables(setup.model),
    age_breaks=nothing,
)
    # -------------------------------------------------------
    # Validate output requests
    # -------------------------------------------------------

    years = collect(output_years)

    isempty(years) &&
        throw(ArgumentError("output_years must not be empty"))

    all(y -> y isa Integer, years) ||
        throw(ArgumentError(
            "output_years must contain integer calendar years"
        ))

    length(unique(years)) == length(years) ||
        throw(ArgumentError("Duplicate output years"))

    t0, t1 = setup.problem.tspan

    all(y -> t0 <= y && y + 1 <= t1, years) ||
        throw(ArgumentError(
            "Output years must lie within the simulation interval"
        ))

    isempty(variables) &&
        throw(ArgumentError("No outputs requested"))

    all(v -> v isa Symbol, variables) ||
        throw(ArgumentError(
            "Output variables must be Symbols"
        ))

    length(unique(variables)) == length(variables) ||
        throw(ArgumentError("Duplicate output variables"))

    # -------------------------------------------------------
    # Allocate one output slot per parameter draw
    # -------------------------------------------------------

    parameter_draws = collect(draws)
    n = length(parameter_draws)

    outputs = Vector{Union{Nothing,NamedTuple}}(
        undef, n
    )
    fill!(outputs, nothing)

    failures = NamedTuple{
        (:draw, :error_type, :message),
        Tuple{Int,String,String}
    }[]

    # -------------------------------------------------------
    # Independent parameter evaluations
    # -------------------------------------------------------

    for i in eachindex(parameter_draws)
        try
            draw = parameter_draws[i]

            params = make_parameters(
                setup.model, draw
            )

            sol = simulate(setup, params)

            successful_retcode(sol) ||
                error(
                    "ODE solver returned $(sol.retcode)"
                )

            out = model_outputs(
                setup.model,
                sol;
                variables,
                years,
                age_breaks,
            )

            # Reject nonfinite predictions rather than
            # silently including them in summaries.
            for variable in variables
                values = getproperty(out, variable)

                all(isfinite, values) ||
                    throw(ArgumentError(
                        "Nonfinite predictions for $variable"
                    ))
            end

            outputs[i] = out

        catch err
            err isa InterruptException && rethrow()

            push!(failures, (
                draw=i,
                error_type=string(typeof(err)),
                message=sprint(showerror, err),
            ))
        end
    end

    return (
        draws=parameter_draws,
        outputs=outputs,
        failures=failures,
    )
end


"""
    simulate_ensemble(model, draws, demography;
                      start_year, end_year,
                      output_years, variables, ...)

Convenience interface that prepares the simulation once,
then evaluates all parameter draws.

Additional keywords configure prepare_simulation.
"""
function simulate_ensemble(
    model::AbstractTBModel,
    draws,
    demography::AbstractDemography;
    start_year::Real,
    end_year::Real,
    output_years,
    variables::Tuple=default_output_variables(model),
    age_breaks=nothing,
    kwargs...,
)
    setup = prepare_simulation(
        model,
        demography;
        start_year,
        end_year,
        kwargs...,
    )

    return simulate_ensemble(
        setup,
        draws;
        output_years,
        variables,
        age_breaks,
    )
end
