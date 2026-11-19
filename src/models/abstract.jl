
"""
    AbstractTBModel

Abstract supertype for deterministic, age-structured TB models.

Every concrete model must implement:

    state_layout(model)
    make_parameters(model, draw::NamedTuple)
    initial_state(model, params, demography; year)
    tb_rhs!(du, u, context, t)

The core model interface does not prescribe parameter types,
priors, likelihoods or epidemiological output definitions.
"""
abstract type AbstractTBModel end


"""
    StateLayout(nepi, ncum, nages)

State-vector layout for an age-structured model.

- `nepi`: number of living epidemiological compartments per age
- `ncum`: number of cumulative event counters per age
- `nages`: number of age classes

Within each age, the living compartments appear first,
followed by the cumulative counters.

State vectors are ordered by age, with compartments varying
fastest. For example:

    age 1: [living..., counters...]
    age 2: [living..., counters...]
    ...

Counter variables record accumulated events. They do not
contribute to the living population.
"""
struct StateLayout
    nepi::Int
    ncum::Int
    nages::Int

    function StateLayout(nepi, ncum, nages)
        nepi > 0 ||
            throw(ArgumentError("nepi must be positive"))

        ncum >= 0 ||
            throw(ArgumentError("ncum must be nonnegative"))

        nages > 0 ||
            throw(ArgumentError("nages must be positive"))

        new(nepi, ncum, nages)
    end
end


"""
Number of state variables per age.
"""
@inline nstates(layout::StateLayout) =
    layout.nepi + layout.ncum


"""
Total state-vector dimension.
"""
@inline state_dimension(layout::StateLayout) =
    nstates(layout) * layout.nages


"""
    state_index(layout, age, compartment)

Return the linear index for the specified age and compartment.

Both indices are one-based. This is a low-level indexing
utility; callers must supply valid indices.
"""
@inline state_index(
    layout::StateLayout,
    age::Int,
    compartment::Int,
) = (age - 1) * nstates(layout) + compartment


# ------------------------------------------------------------
# Required model interface
# ------------------------------------------------------------

"""
    state_layout(model::AbstractTBModel)

Return the model's StateLayout.
"""
function state_layout end


"""
    make_parameters(model::AbstractTBModel, draw::NamedTuple)

Construct model-specific numerical parameters.

Implementations should preserve differentiable numerical
types, including ForwardDiff.Dual, without forced conversion
to Float64.
"""
function make_parameters end


"""
    initial_state(model, params, demography; year)

Construct the complete initial state vector.

The returned vector must:
- have length state_dimension(state_layout(model));
- contain finite numerical values;
- preserve parameter-dependent numerical types.

Initialisation of cumulative event counters is a
model-specific responsibility.
"""
function initial_state end


"""
    tb_rhs!(du, u, context, t)

Evaluate the model's continuous-time differential equations.

The function must:
- populate every element of du;
- leave u unchanged;
- use the supplied SimulationContext;
- preserve differentiable numerical types.

Demographic updates are handled separately by the simulation
infrastructure.
"""
function tb_rhs! end


"""
    priors(model)

Optional model-specific default prior distributions.

This is not required for deterministic simulation.
"""
function priors end


# ------------------------------------------------------------
# Interface validation
# ------------------------------------------------------------

"""
    validate_model_interface(model, params, demography; year)

Perform a basic structural and numerical check of a concrete
model's deterministic simulation interface.

This validates initialisation and one RHS evaluation, but does
not establish scientific correctness, conservation or AD
compatibility.
"""
function validate_model_interface(
    model::AbstractTBModel,
    params,
    demography;
    year::Real,
)
    layout = state_layout(model)

    layout isa StateLayout ||
        throw(ArgumentError(
            "state_layout must return StateLayout"
        ))

    # Initialisation
    u0 = initial_state(
        model, params, demography; year
    )

    u0 isa AbstractVector{<:Real} ||
        throw(ArgumentError(
            "initial_state must return a real-valued vector"
        ))

    length(u0) == state_dimension(layout) ||
        throw(DimensionMismatch(
            "Initial state dimension does not match StateLayout"
        ))

    all(isfinite, u0) ||
        throw(ArgumentError(
            "Initial state contains nonfinite values"
        ))

    # RHS evaluation
    workspace = TBWorkspace(eltype(u0), layout.nages)

    context = SimulationContext(
        model, params, demography, workspace
    )

    du = similar(u0)
    fill!(du, zero(eltype(du)))

    original = copy(u0)

    tb_rhs!(du, u0, context, year)

    all(isfinite, du) ||
        throw(ArgumentError(
            "RHS contains nonfinite derivatives"
        ))

    isequal(u0, original) ||
        throw(ArgumentError(
            "RHS modified the input state"
        ))

    return true
end
