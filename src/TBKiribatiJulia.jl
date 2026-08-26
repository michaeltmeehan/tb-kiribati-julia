module TBKiribatiJulia

include("demography.jl")
include("parameters.jl")
include("model.jl")

export fertility_rate, mortality_rate, migration_flow
export get_population, initial_state, _apply_demography!, apply_demography!, get_age_distribution
export tb_rhs!

end # module
