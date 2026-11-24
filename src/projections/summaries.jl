using DataFrames

function summarise_projections(
    results,
    years;
    age_values=nothing,
    parameter_draw::Int=1,
    demographic_draw::Int=1,
)
    national = DataFrame()

    for (scenario, result) in pairs(results)

        o = result.outputs

        N = vec(sum(o.population; dims=1))
        I = vec(sum(o.incidence; dims=1))
        C = vec(sum(o.notifications; dims=1))
        D = vec(sum(o.mortality; dims=1))
        L = vec(sum(o.latent_prevalence; dims=1))
        A = vec(sum(o.active_prevalence; dims=1))

        ages = isnothing(age_values) ?
               collect(0:(size(o.population, 1)-1)) :
               age_values

        length(ages) == size(o.population, 1) ||
            throw(ArgumentError("Incompatible age dimensions"))

        N65 = vec(sum(
            o.population[ages .>= 65, :];
            dims=1,
        ))

        append!(national, DataFrame(
            scenario=fill(scenario, length(years)),
            parameter_draw=fill(parameter_draw, length(years)),
            demographic_draw=fill(demographic_draw, length(years)),
            year=collect(years),
            population=N,
            proportion_65_plus=N65 ./ N,
            incidence=I,
            notifications=C,
            mortality=D,
            incidence_rate=1e5 .* I ./ N,
            notification_rate=1e5 .* C ./ N,
            mortality_rate=1e5 .* D ./ N,
            latent_prevalence=L ./ N,
            active_prevalence=1e5 .* A ./ N,
        ))
    end

    return national
end


function compare_projections(
    national::AbstractDataFrame;
    variables=(
        :population,
        :incidence,
        :notifications,
        :mortality,
        :incidence_rate,
        :notification_rate,
        :mortality_rate,
    ),
)
    comparisons = DataFrame()

    keys = [:parameter_draw, :demographic_draw, :year]

    contrasts = (
        (:population_growth, :growth_only, :fixed),
        (:demographic_composition, :projected, :growth_only),
        (:combined_demography, :projected, :fixed),
    )

    for (effect, numerator, denominator) in contrasts

        lhs = select(
            filter(r -> r.scenario == numerator, national),
            keys...,
            variables...,
        )

        rhs = select(
            filter(r -> r.scenario == denominator, national),
            keys...,
            variables...,
        )

        paired = innerjoin(
            lhs,
            rhs;
            on=keys,
            renamecols="_numerator" => "_denominator",
            validate=(true, true),
        )

        nrow(paired) == nrow(lhs) == nrow(rhs) ||
            error("Unmatched projection scenarios")

        for variable in variables

            x = paired[!, Symbol(variable, "_numerator")]
            y = paired[!, Symbol(variable, "_denominator")]

            ratio = [
                b == 0 ? missing : a / b
                for (a, b) in zip(x, y)
            ]

            append!(comparisons, DataFrame(
                parameter_draw=paired.parameter_draw,
                demographic_draw=paired.demographic_draw,
                year=paired.year,
                effect=fill(effect, nrow(paired)),
                variable=fill(variable, nrow(paired)),
                numerator=x,
                denominator=y,
                difference=x .- y,
                ratio=ratio,
                percentage_change=100 .* (ratio .- 1),
            ))
        end
    end

    return sort!(
        comparisons,
        [:parameter_draw, :demographic_draw,
            :year, :variable, :effect],
    )
end