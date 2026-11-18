
include("kernel.jl")
include("indices.jl")

struct RagonnetTBModel <: AbstractTBModel
    layout::StateLayout
    contact::Matrix{Float64}
end

function RagonnetTBModel(
    contact::AbstractMatrix{<:Real}
)
    size(contact) == (96, 96) ||
        throw(DimensionMismatch(
            "Ragonnet currently requires a 96×96 contact matrix"
        ))

    all(x -> isfinite(x) && x >= 0, contact) ||
        throw(ArgumentError(
            "Contact matrix must be finite and nonnegative"
        ))

    return RagonnetTBModel(
        StateLayout(10, 7, 96),
        Matrix{Float64}(contact),
    )
end

state_layout(model::RagonnetTBModel) = model.layout

include("parameters.jl")
include("dynamics.jl")

function tb_rhs!(
    du,
    u,
    ctx::SimulationContext{<:RagonnetTBModel},
    t,
)
    return ragonnet_rhs!(
        du,
        u,
        ctx.params,
        t,
        state_layout(ctx.model),
        ctx.workspace,
    )
end
