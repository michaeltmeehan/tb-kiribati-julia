
using SciMLBase: successful_retcode

"""
    simulate_ensemble(
        model::AbstractTBModel,
        draws,
        demography::AbstractDemography;
        start_year,
        end_year,
        output_years,
        age_breaks=nothing,
        kwargs...,
    )

Simulate an ensemble of parameter draws and extract annual
epidemiological outputs.

Each draw is processed independently through:
    make_parameters -> initial_state -> simulate -> model_outputs

Failed simulations are recorded without resampling.

Returns a NamedTuple containing:
- draws: original parameter draws
- outputs: model outputs, or `nothing` for failed draws
- failures: error records identifying unsuccessful draws

Additional keyword arguments are forwarded to simulate_from_draw.
"""
function simulate_ensemble(
    model::AbstractTBModel,
    draws,
    demography::AbstractDemography;
    start_year::Real,
    end_year::Real,
    output_years,
    age_breaks=nothing,
    kwargs...,
)
    start_year < end_year ||
        throw(ArgumentError(
            "start_year must precede end_year"
        ))

    years = collect(output_years)

    isempty(years) &&
        throw(ArgumentError("output_years must not be empty"))

    all(y -> start_year <= y && y + 1 <= end_year, years) ||
        throw(ArgumentError(
            "Output years must lie within the simulation period"
        ))

    # Preserve the complete collection of parameter draws.
    parameter_draws = collect(draws)
    n = length(parameter_draws)

    # Retain one output slot per draw, including failures.
    outputs = Vector{Union{Nothing,NamedTuple}}(undef, n)
    fill!(outputs, nothing)

    failures = NamedTuple{
        (:draw, :error_type, :message),
        Tuple{Int,String,String}
    }[]

    for i in eachindex(parameter_draws)

        try
            # Construct parameters, initialise, and simulate.
            sol = simulate_from_draw(
                model,
                parameter_draws[i],
                demography;
                start_year=start_year,
                end_year=end_year,
                kwargs...,
            )

            successful_retcode(sol.retcode) ||
                error("ODE solver returned $(sol.retcode)")

            # Extract model-specific outputs.
            out = model_outputs(
                model,
                sol;
                years=years,
                age_breaks=age_breaks,
            )

            # Only store the output after successful extraction.
            outputs[i] = out

        catch err

            # Allow the user to interrupt long ensembles.
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
