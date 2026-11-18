
module RagonnetKernel

using DelimitedFiles
using LinearAlgebra: mul!

# Legacy source dependencies.
data_path(name) = normpath(
    joinpath(@__DIR__, "..", "..", "..", "data", name)
)

# These are snapshots of the original source, not
# references to files that we intend to delete.
include("legacy_parameters.jl")
include("legacy_dynamics.jl")

end
