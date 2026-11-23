import ForwardDiff

function initial_cohort_generator(
    params::DoddExtendedTBParameters,
    a::Int,
    year::Real,
)
    T = typeof(params.beta)
    Q = zeros(T, 9, 9)

    # Auxiliary state indices
    S, E, L = 1, 2, 3
    ISP, ISN, IEP = 4, 5, 6
    TR, R, D = 7, 8, 9

    # Column-oriented transition generator.
    function transition!(from, to, rate)
        Q[to, from] += rate
        Q[from, from] -= rate
    end

    # Primary infection and progression.
    transition!(S, E, params.Lambda0)
    transition!(E, L, params.kappa[a])

    for k in 1:3
        I = ISP + k - 1
        transition!(
            E, I,
            params.epsilon[a] * params.presentation_fast[a, k]
        )

        transition!(
            L, I,
            params.nu[a] * params.presentation_slow[a, k]
        )
    end

    # Detection probability translated to the
    # reference detection-hazard scale.
    pd = detection_probability(params.detection, year)
    odds = pd / (one(T) - pd)

    for k in 1:3
        I = ISP + k - 1

        δ = odds *
            params.detection_multiplier[a] *
            params.relative_detection[k]

        transition!(I, TR, δ)

        transition!(
            I, L, params.self_cure[a, k]
        )

        transition!(
            I, D, params.tb_mortality[a, k]
        )
    end

    # Treatment completion and death.
    transition!(
        TR, R,
        params.treatment_rate *
            (one(T) - params.treatment_cfr)
    )

    transition!(
        TR, D,
        params.treatment_rate *
            params.treatment_cfr
    )

    return Q
end

"""
Propagate a cohort through a constant CTMC generator.

Uses uniformisation to evaluate exp(Q * duration) * x
without requiring a matrix exponential of Dual numbers.

Q uses column-oriented transition rates.
"""
function propagate_initial_cohort(
    x::AbstractVector,
    Q::AbstractMatrix,
    duration::Real;
    tol::Real=1e-13,
)
    n = length(x)

    size(Q) == (n, n) ||
        throw(DimensionMismatch("Invalid generator dimensions"))

    duration >= 0 ||
        throw(ArgumentError("Negative propagation duration"))

    # Use primal values only to select the numerical
    # uniformisation rate and number of substeps.
    rate = maximum(
        -ForwardDiff.value(Q[i, i]) for i in 1:n
    )

    rate >= 0 ||
        throw(ArgumentError("Invalid generator"))

    if rate == 0 || duration == 0
        return copy(x)
    end

    # Keep the Poisson mean modest for numerical stability.
    nsteps = max(1, ceil(Int, rate * duration / 4))
    h = duration / nsteps
    z = rate * h

    T = promote_type(eltype(x), eltype(Q))
    P = Matrix{T}(Q ./ rate)

    for i in 1:n
        P[i, i] += one(T)
    end

    result = T.(x)

    for _ in 1:nsteps
        term = copy(result)
        weight = exp(-z)

        next = weight .* term
        cumulative = weight

        for k in 1:200
            term = P * term
            weight *= z / k

            next .+= weight .* term
            cumulative += weight

            if 1 - cumulative <= tol
                break
            end

            k == 200 && throw(ErrorException(
                "Uniformisation series did not converge"
            ))
        end

        result = next
    end

    return result
end

"""
Construct the DoddExtended initial state directly.

The initial distribution follows the same heuristic
previously used through the Dodd reference model.

All initial active TB is treatment-history 0.
Recovered individuals occupy the common R compartment.
"""

function initial_state(
    model::DoddExtendedTBModel,
    params::DoddExtendedTBParameters,
    demography::AbstractDemography;
    year::Real,
)
    layout = state_layout(model)
    nages = layout.nages
    stride = nstates(layout)

    population = get_population(demography, year)

    length(population) == nages ||
        throw(DimensionMismatch(
            "Population length does not match model"
        ))

    all(v -> isfinite(v) && v >= 0, population) ||
        throw(ArgumentError("Invalid population"))

    T = promote_type(
        eltype(population),
        typeof(params.beta),
    )

    u0 = zeros(T, stride * nages)

    # Birth cohort initially entirely susceptible.
    x = zeros(T, 9)
    x[1] = one(T)

    Qprevious = nothing

    for a in 1:nages
        Q = initial_cohort_generator(
            params, a, year
        )

        # Approximate age-a population at midpoint
        # of the corresponding one-year age interval.
        if a == 1
            x = propagate_initial_cohort(
                x, Q, 0.5
            )
        else
            x = propagate_initial_cohort(
                x, Qprevious, 0.5
            )
            x = propagate_initial_cohort(
                x, Q, 0.5
            )
        end

        Qprevious = Q

        # Condition on not having died from TB.
        survival = sum(@view x[1:8])

        survival > 0 ||
            throw(ArgumentError(
                "No surviving cohort at age $(a-1)"
            ))

        z = population[a] / survival
        base = (a - 1) * stride

        u0[base+DE.S] = z * x[1]
        u0[base+DE.E0] = z * x[2]
        u0[base+DE.L0] = z * x[3]

        u0[base+DE.I0SP] = z * x[4]
        u0[base+DE.I0SN] = z * x[5]
        u0[base+DE.I0EP] = z * x[6]

        u0[base+DE.T] = z * x[7]
        u0[base+DE.R] = z * x[8]
    end

    return u0
end
