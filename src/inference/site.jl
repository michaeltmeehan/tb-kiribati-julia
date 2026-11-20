using Distributions: Binomial, logpdf

"""
    SiteTBObservation(rows)

Annual pulmonary and extrapulmonary notifications, conditional on their
reported site-classified total. `rows` may be returned by
`load_site_notifications`; incomplete or unclassified totals are not imputed.
"""
struct SiteTBObservation{R}
    rows::R
end

function SiteTBObservation(rows::AbstractVector)
    clean = NamedTuple[]
    for r in rows
        year = Int(r.year)
        pulmonary = Int(r.pulmonary)
        extrapulmonary = Int(r.extrapulmonary)
        total = Int(r.total)

        (pulmonary >= 0 && extrapulmonary >= 0 &&
            pulmonary + extrapulmonary == total) ||
            throw(ArgumentError("Invalid site counts in $year"))

        push!(clean, (; year, pulmonary, extrapulmonary, total))
    end

    length(unique(r.year for r in clean)) == length(clean) ||
        throw(ArgumentError("Duplicate site observation year"))

    sort!(clean; by=r -> r.year)
    return SiteTBObservation{typeof(clean)}(clean)
end

site_loglikelihood(::Nothing, output) = 0.0

"""
Conditional binomial likelihood for the pulmonary fraction among
site-classified notifications. The site total is NOT another count likelihood.
"""
function site_loglikelihood(obs::SiteTBObservation, output)
    hasproperty(output, :pulmonary_notifications) &&
        hasproperty(output, :extrapulmonary_notifications) ||
        throw(ArgumentError(
            "Model must provide both site notification outputs"
        ))

    pulmonary = output.pulmonary_notifications
    extrapulmonary = output.extrapulmonary_notifications

    size(pulmonary) == size(extrapulmonary) ||
        throw(DimensionMismatch("Site output dimensions differ"))

    ll = 0.0

    for r in obs.rows
        j = findfirst(==(r.year), output.years)
        isnothing(j) &&
            throw(ArgumentError("Missing site prediction year $(r.year)"))

        predicted_pulmonary = sum(@view pulmonary[:, j])
        predicted_extrapulmonary = sum(@view extrapulmonary[:, j])
        total = predicted_pulmonary + predicted_extrapulmonary

        if !(isfinite(predicted_pulmonary) &&
            isfinite(predicted_extrapulmonary) &&
            predicted_pulmonary >= 0 &&
            predicted_extrapulmonary >= 0 &&
            total > 0)
            return -Inf
        end

        ll += logpdf(
            Binomial(
                r.total,
                predicted_pulmonary / total,
            ),
            r.pulmonary,
        )
    end

    return ll
end