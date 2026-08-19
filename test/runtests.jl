using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Test
using TBKiribatiJulia
include("test_model.jl")
include("test_contacts.jl")
include("test_demography.jl")
include("test_wpp_demography.jl")
