module TBKiribatiJulia

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))

data_path(filename) = joinpath(DATA_DIR, filename)

include("demography.jl")
include("parameters.jl")
include("model.jl")
include("equilibrium_demography.jl")
include("simulation.jl")
include("incidence.jl")
include("summary.jl")
include("comparison.jl")

export fertility_rate, mortality_rate, migration_flow
export get_population, seeded_initial_state, _apply_demography!, apply_demography!, get_age_distribution, aggregate_age_bands
export tb_rhs!
export _apply_equilibrium_demography!
export CONTACT, EQUILIBRIUM_YEAR
export reset_cumulative_counters!, make_annual_callback, simulate
export raw_annual_incidence_by_age, raw_annual_deaths_by_age, annual_incidence_per_100k, aggregate_age_groups, median_age, demographic_summary
export simulation_summary
export with_tb_mortality
export run_demography_comparison, comparison_summary

end # module
