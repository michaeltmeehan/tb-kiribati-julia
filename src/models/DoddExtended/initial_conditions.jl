
"""
Map a Dodd initial state into the extended state space.

Initial active disease has treatment history 0.
Existing recovered patients have treatment history 1.
"""
function extend_initial_state(
    model::DoddExtendedTBModel,
    original::AbstractVector,
)
    old_layout = state_layout(model.reference)
    new_layout = state_layout(model)

    nages = new_layout.nages
    old_stride = nstates(old_layout)
    new_stride = nstates(new_layout)

    length(original) == nages * old_stride ||
        throw(DimensionMismatch(
            "Incorrect original state length"
        ))

    u0 = zeros(Float64, nages * new_stride)

    @inbounds for a in 1:nages
        old = (a - 1) * old_stride
        new = (a - 1) * new_stride

        u0[new+DE.S] =
            original[old+S]

        u0[new+DE.E0] =
            original[old+E]

        u0[new+DE.L0] =
            original[old+L]

        for k in 1:3
            u0[new+DE.I0SP+k-1] =
                model.presentation[a, k] *
                    original[old+I]
        end

        u0[new+DE.T] =
            original[old+T]

        u0[new+DE.R] =
            original[old+R]
    end

    return u0
end


function initial_state(
    model::DoddExtendedTBModel,
    params::DoddExtendedTBParameters,
    demography::AbstractDemography;
    year::Real,
)
    original = initial_state(
        model.reference,
        params.base,
        demography;
        year=year,
    )

    return extend_initial_state(model, original)
end
