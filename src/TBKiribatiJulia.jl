module TBKiribatiJulia

include("demography.jl")
include("parameters.jl")
include("model.jl")
include("wpp_demography.jl")

export WPP_COUNTRY_KIRIBATI, WPP_YEARS, WPP_OPEN_AGE
export load_kiribati_wpp_data, load_kiribati_agepi_reference
export wpp_kiribati_population_2025, wpp_kiribati_demographic_schedule

end # module
