using Random
using Distributions: Binomial
using DataFrames

"""
    conditional_binomial_predictive(
        draws, observations, probability; rng
    )

Generate replicated binomial observations conditional
on observed denominators and model-predicted probabilities.

Observations must contain year, numerator and denominator.
"""
function conditional_binomial_predictive(
    draws::AbstractDataFrame,
    observations::AbstractDataFrame,
    probability::Symbol;
    rng=Random.default_rng(),
)
    obs = Dict(
        r.year => r for r in eachrow(observations)
    )

    rows = NamedTuple[]

    for r in eachrow(draws)
        haskey(obs, r.year) || continue

        o = obs[r.year]
        p = r[probability]
        n = Int(o.denominator)

        isfinite(p) && 0 <= p <= 1 || continue
        n > 0 || continue

        replicated = rand(rng, Binomial(n, p))

        push!(rows, (
            draw=r.draw,
            year=r.year,
            probability=p,
            numerator=o.numerator,
            denominator=n,
            expected_count=n*p,
            predictive_count=replicated,
            expected_fraction=p,
            predictive_fraction=replicated/n,
            observed_fraction=o.numerator/n,
        ))
    end

    return DataFrame(rows)
end

using Distributions: NegativeBinomial

"""
    notification_predictive(annual, age; sigma_obs, tau_year, rng)

Generate replicated notification observations using
the shared annual random-effect NB2 observation model.

Annual and age-specific predictions must contain
draw_id and year columns.
"""
function notification_predictive(
    annual::AbstractDataFrame,
    age::AbstractDataFrame;
    sigma_obs=0.30,
    tau_year=0.18,
    rng=Random.default_rng(),
)
    phi = inv(sigma_obs^2)

    function replicate(mu)
        mu = max(0.0, mu)

        return mu == 0 ? 0 : rand(
            rng,
            NegativeBinomial(phi, phi / (phi + mu)),
        )
    end

    # Index age-specific predictions by draw and year
    age_index = Dict(
        (g.draw_id[1], g.year[1]) => g
        for g in groupby(age, [:draw_id, :year])
    )

    annual_rows = NamedTuple[]
    age_rows = NamedTuple[]

    for r in eachrow(annual)

        key = (r.draw_id, r.year)

        # Shared annual observation effect
        m = exp(
            tau_year * randn(rng) - tau_year^2 / 2
        )

        if haskey(age_index, key)

            g = age_index[key]

            noisy = [
                replicate(m * a.expected)
                for a in eachrow(g)
            ]

            for (a, y) in zip(eachrow(g), noisy)
                push!(age_rows, (
                    draw=r.draw_id,
                    year=r.year,
                    age_group=a.age_group,
                    observed=a.observed,
                    expected=a.expected,
                    predictive=y,
                    annual_multiplier=m,
                ))
            end

            expected = sum(g.expected)
            predictive = sum(noisy)
            observed = sum(g.observed)
            observation_type = "age-disaggregated"

        else

            expected = r.notifications
            predictive = replicate(m * expected)
            observed = missing
            observation_type = "national"

        end

        push!(annual_rows, (
            draw=r.draw_id,
            year=r.year,
            expected=expected,
            predictive=predictive,
            observed=observed,
            annual_multiplier=m,
            observation_type=observation_type,
        ))
    end

    return (
        annual=DataFrame(annual_rows),
        age=DataFrame(age_rows),
    )
end