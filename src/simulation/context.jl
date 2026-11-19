
"""
    TBWorkspace{T}

Reusable age-indexed arrays for force-of-infection calculations.
The numerical type T may be Float64, BigFloat, or an AD number type.
"""
struct TBWorkspace{T<:Real}
    foi::Vector{T}
    infectious_prevalence::Vector{T}
end

function TBWorkspace(
    ::Type{T},
    nages::Integer,
) where {T<:Real}
    nages > 0 ||
        throw(ArgumentError("nages must be positive"))

    return TBWorkspace{T}(
        zeros(T, nages),
        zeros(T, nages),
    )
end

# Preserve existing usage.
TBWorkspace(nages::Integer) =
    TBWorkspace(Float64, nages)


"""
    SimulationContext

Container for the model, its parameters, demography
and numerical workspace.
"""
struct SimulationContext{
    M<:AbstractTBModel,
    P,
    D,
    W<:TBWorkspace,
}
    model::M
    params::P
    demography::D
    workspace::W
end
