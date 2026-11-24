using SciMLBase: successful_retcode

function project_scenarios(
    model::AbstractTBModel,
    params,
    data::DemographicData;
    u0::AbstractVector,
    start_year::Integer,
    end_year::Integer,
    variables::Tuple=(
        :population,
        :incidence,
        :notifications,
        :mortality,
        :latent_prevalence,
        :active_prevalence,
    ),
    age_breaks=nothing,
    simulation_kwargs...,
)
    start_year <= end_year ||
        throw(ArgumentError("Invalid projection interval"))

    data.first_year <= start_year &&
        end_year + 1 <= data.last_year ||
        throw(ArgumentError("Projection outside demographic data"))

    nages = state_layout(model).nages

    scenarios = (
        fixed=EquilibriumDemography(
            data, start_year; nages
        ),
        growth_only=GrowthOnlyDemography(
            data, start_year; nages
        ),
        projected=ProjectedDemography(
            data; nages
        ),
    )

    years = collect(start_year:end_year)

    results = map(scenarios) do demography
        setup = prepare_simulation(
            model,
            demography;
            start_year,
            end_year=end_year + 1,
            simulation_kwargs...,
        )

        sol = simulate(setup, params; u0=copy(u0))

        successful_retcode(sol) ||
            error("Projection failed: $(sol.retcode)")

        outputs = model_outputs(
            model,
            sol;
            variables,
            years,
            age_breaks,
        )

        return (
            demography=demography,
            solution=sol,
            outputs=outputs,
        )
    end

    return results
end