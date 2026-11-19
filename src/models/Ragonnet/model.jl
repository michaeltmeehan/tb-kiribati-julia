
include("indices.jl")

struct RagonnetTBModel <: AbstractTBModel
    layout::StateLayout
    contact::Matrix{Float64}
end

function RagonnetTBModel(
    contact::AbstractMatrix{<:Real},
)
    size(contact) == (96, 96) ||
        throw(DimensionMismatch(
            "Ragonnet requires a 96×96 contact matrix"
        ))

    all(x -> isfinite(x) && x >= 0, contact) ||
        throw(ArgumentError(
            "Contact matrix must be finite and nonnegative"
        ))

    # Preserve the original contact scale.
    # Unlike Dodd, Ragonnet does not normalise contacts.
    return RagonnetTBModel(
        StateLayout(10, 7, 96),
        Matrix{Float64}(contact),
    )
end

state_layout(model::RagonnetTBModel) = model.layout

include("parameters.jl")
include("dynamics.jl")