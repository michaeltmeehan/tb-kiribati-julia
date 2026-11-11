abstract type AbstractTBModel end

struct StateLayout
    nepi::Int
    ncum::Int
    nages::Int

    function StateLayout(nepi, ncum, nages)
        nepi > 0 || throw(ArgumentError("nepi must be positive"))
        ncum >= 0 || throw(ArgumentError("ncum must be nonnegative"))
        nages > 0 || throw(ArgumentError("nages must be positive"))
        new(nepi, ncum, nages)
    end
end

@inline nstates(l::StateLayout) = l.nepi + l.ncum

@inline state_index(l::StateLayout, age::Int, compartment::Int) =
    (age - 1) * nstates(l) + compartment

function state_layout end
function tb_rhs! end
function initial_state end
function priors end