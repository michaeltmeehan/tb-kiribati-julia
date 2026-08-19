using Test
using DelimitedFiles: readdlm

const CONTACT_REFERENCE_DIR = joinpath(@__DIR__, "..", "validation", "reference", "contact")

function _read_numeric_reference(path::AbstractString)
    return Matrix{Float64}(readdlm(path, ',', Float64))
end

@testset "Solomon Islands contact source" begin
    source = load_solomon_islands_contact_source()
    @test source.metadata.source == :prem
    @test source.metadata.country == :solomon_islands
    @test source.metadata.setting == :all
    @test source.metadata.orientation == :recipient_source
    @test length(source.age_bands) == 16
    @test source.age_bands[1].label == "00_05"
    @test source.age_bands[end].label == "75_80"
    @test source.age_bands[end].lower == 75
    @test isinf(source.age_bands[end].upper)
    @test size(source.matrix) == (16, 16)
    @test all(isfinite, source.matrix)
    @test minimum(source.matrix) ≥ 0.0
    @test !isapprox(source.matrix, transpose(source.matrix))
end

@testset "Source-band expansion" begin
    source = load_solomon_islands_contact_source()
    empirical = load_solomon_islands_contact_matrix()
    reference = _read_numeric_reference(joinpath(CONTACT_REFERENCE_DIR, "prem_solomon_islands_all_age96_agepi.csv"))

    @test size(empirical) == (96, 96)
    @test size(reference) == (96, 96)
    @test all(isfinite, empirical)
    @test minimum(empirical) ≥ 0.0
    @test maximum(abs.(empirical .- reference)) ≤ 1e-12

    @test isapprox(empirical[1, 1], source.matrix[1, 1] / 5; atol = 1e-12, rtol = 1e-12)
    @test isapprox(empirical[5, 5], source.matrix[1, 1] / 5; atol = 1e-12, rtol = 1e-12)
    @test isapprox(empirical[6, 6], source.matrix[2, 2] / 5; atol = 1e-12, rtol = 1e-12)
    @test isapprox(empirical[76, 76], source.matrix[16, 16] / 21; atol = 1e-12, rtol = 1e-12)
    @test isapprox(empirical[96, 20], source.matrix[16, 4] / 5; atol = 1e-12, rtol = 1e-12)
    @test isapprox(empirical[20, 96], source.matrix[4, 16] / 21; atol = 1e-12, rtol = 1e-12)
    @test isapprox(empirical[95, 95], source.matrix[16, 16] / 21; atol = 1e-12, rtol = 1e-12)

    @test isapprox(empirical[1, 1], empirical[2, 1]; atol = 1e-12, rtol = 1e-12)
    @test isapprox(empirical[76, 1], empirical[77, 1]; atol = 1e-12, rtol = 1e-12)

    @test isapprox(sum(empirical[1, 1:5]), source.matrix[1, 1]; atol = 1e-12, rtol = 1e-12)
    @test isapprox(sum(empirical[76, 76:96]), source.matrix[16, 16]; atol = 1e-12, rtol = 1e-12)
end

@testset "Orientation" begin
    source_matrix = zeros(Float64, 16, 16)
    source_matrix[4, 5] = 7.0
    source_matrix[5, 4] = 11.0
    source_matrix[16, 4] = 13.0
    source_matrix[4, 16] = 17.0
    source_bands = load_solomon_islands_contact_source().age_bands
    target = adapt_source_band_contact_matrix(source_matrix, source_bands)

    p = zero_transition_params(; beta = 1.0)
    p.contact .= target
    u = blank_state()
    U = reshape(u, NSTATE, NAGE)
    U[SubClinLow, 17] = 1.0
    U[SubClinInf, 17] = 2.0
    U[ClinLow, 17] = 3.0
    U[ClinInf, 17] = 4.0
    U[SubClinLow, 21] = 5.0
    U[SubClinInf, 21] = 6.0
    U[ClinLow, 21] = 7.0
    U[ClinInf, 21] = 8.0
    U[SubClinLow, 96] = 9.0
    U[SubClinInf, 96] = 10.0
    U[ClinLow, 96] = 11.0
    U[ClinInf, 96] = 12.0
    U[MtbNaive, 17] = 100.0
    U[MtbNaive, 21] = 100.0
    U[MtbNaive, 96] = 100.0

    λ = zeros(Float64, NAGE)
    compute_force_of_infection!(λ, u, p)

    q4 = (0.2 * 1 + 0.5 * 2 + 0.4 * 3 + 1.0 * 4) / 110.0
    q5 = (0.2 * 5 + 0.5 * 6 + 0.4 * 7 + 1.0 * 8) / 126.0
    q16 = (0.2 * 9 + 0.5 * 10 + 0.4 * 11 + 1.0 * 12) / 142.0
    expected16 = 7.0 * q5 / 5 + 17.0 * q16 / 21
    expected20 = 11.0 * q4 / 5
    @test isapprox(λ[17], expected16; atol = 1e-12, rtol = 1e-12)
    @test isapprox(λ[21], expected20; atol = 1e-12, rtol = 1e-12)
    @test λ[17] != λ[21]
end
