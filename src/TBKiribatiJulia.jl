module TBKiribatiJulia

include("demography.jl")
include("parameters.jl")
include("model.jl")
include("static_demography.jl")
include("simulation.jl")
include("incidence.jl")

export fertility_rate, mortality_rate, migration_flow
export get_population, initial_state, _apply_demography!, apply_demography!, get_age_distribution, aggregate_age_bands
export tb_rhs!
export _apply_static_demography!
export CONTACT, STATIC_YEAR
export reset_cumulative_counters!, make_annual_callback
export raw_annual_incidence_by_age, annual_incidence_per_100k

end # module
