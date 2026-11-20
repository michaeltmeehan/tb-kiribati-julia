
using Distributions: Binomial, logpdf

"""
Conditional binomial observation model for deaths among
registered treatment cohorts.

By default, exclude 2024 because WHO changed the
treatment-outcome reporting definition.
"""
struct TreatmentDeathObservation{R}
    rows::R
    treatment_duration::Float64
    background_hazard::Float64
end

function TreatmentDeathObservation(
    rows;
    include_2024::Bool=false,
    treatment_duration::Real=0.5,
    background_hazard::Real=0.0,
)
    isfinite(treatment_duration) && treatment_duration > 0 ||
        throw(ArgumentError("Invalid treatment duration"))

    isfinite(background_hazard) && background_hazard >= 0 ||
        throw(ArgumentError("Invalid background mortality"))

    selected = [
        r for r in rows
              if include_2024 || r.year != 2024
    ]

    isempty(selected) &&
        throw(ArgumentError("No treatment cohorts selected"))

    length(unique(r.year for r in selected)) ==
        length(selected) ||
        throw(ArgumentError("Duplicate treatment cohort year"))

    for r in selected
        ismissing(r.died) &&
            throw(ArgumentError(
                "Missing deaths in $(r.year)"
            ))

        ismissing(r.cohort_size) &&
            throw(ArgumentError(
                "Missing cohort size in $(r.year)"
            ))

        r.cohort_size > 0 ||
            throw(ArgumentError(
                "Nonpositive cohort size in $(r.year)"
            ))

        0 <= r.died <= r.cohort_size ||
            throw(ArgumentError(
                "Invalid deaths in $(r.year)"
            ))
    end

    return TreatmentDeathObservation(
        selected,
        Float64(treatment_duration),
        Float64(background_hazard),
    )
end


function treatment_death_probability(
    obs::TreatmentDeathObservation,
    theta::NamedTuple,
)
    haskey(theta, :treatment_cfr) ||
        throw(ArgumentError(
            "Missing treatment_cfr parameter"
        ))

    q = theta.treatment_cfr
    0 <= q <= 1 || return NaN

    tau = inv(obs.treatment_duration)
    mu = obs.background_hazard

    return (q * tau + mu) / (tau + mu)
end


treatment_death_loglikelihood(::Nothing, theta) = 0.0

function treatment_death_loglikelihood(
    obs::TreatmentDeathObservation,
    theta::NamedTuple,
)
    p = treatment_death_probability(obs, theta)

    isfinite(p) && 0 <= p <= 1 || return -Inf

    return sum(
        logpdf(
            Binomial(r.cohort_size, p),
            r.died,
        )
        for r in obs.rows
    )
end


function treatment_death_fit_rows(
    obs::TreatmentDeathObservation,
    theta::NamedTuple,
)
    p = treatment_death_probability(obs, theta)

    return [
        (
            year=r.year,
            cohort_size=r.cohort_size,
            observed_deaths=r.died,
            expected_deaths=r.cohort_size * p,
            observed_fraction=r.died / r.cohort_size,
            expected_fraction=p,
        )
        for r in obs.rows
    ]
end
