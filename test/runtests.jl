using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Test
using TBKiribatiJulia
include("test_model.jl")
