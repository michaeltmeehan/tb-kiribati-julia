
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
increment in CumProgressionToActiveTB.
"""
function annual_incidence_per_100k(sol)

    years = sol.t[2:end]

    incidence = Vector{Float64}(undef, length(years))

    for i in 2:length(sol.t)

        cases =
            cumulative_incidence(sol.u[i]) -
            cumulative_incidence(sol.u[i - 1])

        population = sum(get_age_distribution(sol.u[i]))

        incidence[i - 1] = 1e5 * cases / population

    end

    return years, incidence
end