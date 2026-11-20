using Test
using TBKiribatiJulia

include(joinpath(
    @__DIR__, "..", "..",
    "scripts", "common", "kiribati_paths.jl",
))

@testset "Surveillance paths and datasets" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = kiribati_data_paths(root)

    @test isdir(paths.surveillance)
    @test isdir(paths.estimates)
    @test isdir(paths.demography)
    @test isdir(paths.contacts)

    @test isfile(paths.legacy_contact)
    @test isfile(joinpath(
        paths.legacy_demography, "population"
    ))

    age_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+",
    ]

    surveillance = load_kiribati_surveillance(
        paths.surveillance;
        age_labels,
        age_years=2013:2025,
    )

    for field in (
        :age_notifications,
        :national_notifications,
        :recurrent_notifications,
        :site_notifications,
        :treatment_outcomes,
        :treatment_outcomes_all,
        :historical_new_by_type,
    )
        @test !isempty(getproperty(surveillance, field))
    end

    @test !hasproperty(
        surveillance, :historical_new_by_age
    )
end