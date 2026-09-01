module TBKiribatiJulia

include("demography.jl")
include("parameters.jl")
include("model.jl")
include("static_demography.jl")
include("incidence.jl")

export fertility_rate, mortality_rate, migration_flow
export get_population, initial_state, _apply_demography!, apply_demography!, get_age_distribution
export tb_rhs!
export _apply_static_demography!
export CONTACT, STATIC_YEAR
export raw_annual_incidence_by_age, annual_incidence_per_100k

end # module
