
using Test
using CSV
using DataFrames
using LinearAlgebra

let
    root = normpath(joinpath(@__DIR__, "..", ".."))

    include(joinpath(
        root, "src", "data", "processed.jl"
    ))

    api = ProcessedData

    @testset "Processed data interface" begin

        notifications = api.load_surveillance(
            :notifications_total
        )

        @test nrow(notifications) == 45

        age = api.load_surveillance(
            :notifications_by_age;
            years=2023:2025
        )

        @test all(y -> 2023 <= y <= 2025, age.year)

        recurrence = api.load_surveillance(
            :recurrent; years=2013:2024
        )

        @test nrow(recurrence) == 12

        burden = api.load_estimates(
            :burden_national; years=[2025]
        )

        @test nrow(burden) == 1
        @test only(burden.e_inc_num) == 1100

        population = api.load_demography(
            :population; years=[2023]
        )

        @test nrow(population) == 101
        @test isapprox(
            sum(population.pop), 133527;
            atol=1
        )

        C0 = api.load_contacts()
        C1 = api.load_contacts(
            variant=:reciprocal_2023
        )

        @test size(C0) == (96, 96)
        @test size(C1) == (96, 96)
        @test all(C0 .>= 0)
        @test all(C1 .>= 0)
        @test norm(C0 - C1) > 0

        @test_throws ArgumentError api.load_surveillance(
            :nonexistent_dataset
        )

        @test_throws ArgumentError api.load_contacts(
            variant=:nonexistent
        )
    end
end
