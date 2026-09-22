
"""
    cumulative_incidence(u)

Total cumulative progression to active TB across all ages.
"""
function cumulative_incidence(u)
    total = 0.0

    @inbounds for a in 1:NAGE
        base = (a - 1) * NSTATE
        total += u[base + CumProgressionToActiveTB]
    end

    return total
end


function raw_annual_incidence_by_age(sol)

    years = collect(Int(round(sol.t[1])) + 1:Int(round(sol.t[end])))

    incidence = Matrix{Float64}(undef, length(years), NAGE)

    for (i, t) in enumerate(years)
        u = sol(t, continuity = :left)

        @inbounds for a in 1:NAGE
            base = (a - 1) * NSTATE
            incidence[i, a] = u[base + CumProgressionToActiveTB]
        end
    end

    return years, incidence
end


"""
    annual_incidence_per_100k(sol)

Calculate annual TB incidence per 100,000 population from the
CumProgressionToActiveTB counter immediately before each annual reset.
"""
function annual_incidence_per_100k(sol)

    years = collect((sol.prob.tspan[1] + 1.0):1.0:sol.prob.tspan[2])

    incidence = Vector{Float64}(undef, length(years))

    for (i, t) in enumerate(years)

        u = sol(t, continuity = :left)

        cases = cumulative_incidence(u)
        population = sum(get_age_distribution(u))

        incidence[i] = 1e5 * cases / population
    end

    return years, incidence
end


function aggregate_age_groups(values, age_breaks)
    length(values) == NAGE ||
        throw(ArgumentError("values must have length $NAGE"))

    first(age_breaks) == 0 ||
        throw(ArgumentError("age_breaks must start at 0"))

    last(age_breaks) == NAGE ||
        throw(ArgumentError("age_breaks must end at $NAGE"))

    issorted(age_breaks) ||
        throw(ArgumentError("age_breaks must be sorted"))

    return [
        sum(@view values[(lo + 1):hi])
        for (lo, hi) in zip(age_breaks[1:end-1], age_breaks[2:end])
    ]
end