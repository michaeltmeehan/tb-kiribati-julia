module TBKiribatiJulia

include("demography.jl")
include("contacts.jl")
include("parameters.jl")
include("model.jl")
include("wpp_demography.jl")
include("burnin.jl")

export WPP_COUNTRY_KIRIBATI, WPP_YEARS, WPP_OPEN_AGE
export ContactAgeBand
export adapt_source_band_contact_matrix, load_solomon_islands_contact_source, load_solomon_islands_contact_matrix
export load_kiribati_wpp_data, load_kiribati_agepi_reference
export wpp_kiribati_population, wpp_kiribati_population_1950, wpp_kiribati_population_2025, wpp_kiribati_demographic_schedule
export BurnInCheckpoint, BurnInResult, HistoricalInitializationResult, rebase_state_to_population, run_stationary_burnin, generate_historical_initial_state

end # module
