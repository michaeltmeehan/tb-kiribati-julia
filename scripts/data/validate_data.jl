
using Test
using CSV
using DataFrames
using DelimitedFiles
using Statistics
using LinearAlgebra

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    datadir = joinpath(root, "data", "processed")

    survdir = joinpath(datadir, "surveillance")
    demodir = joinpath(datadir, "demography")
    contactdir = joinpath(datadir, "contacts")

    read_surv(name) = CSV.read(
        joinpath(survdir, "$name.csv"), DataFrame
    )

    read_demo(name) = CSV.read(
        joinpath(demodir, "$name.csv"), DataFrame
    )

    function check_unique(df, cols)
        @test nrow(unique(df[:, cols])) == nrow(df)
    end

    function check_counts(df, cols)
        for c in cols
            @test all(x ->
                ismissing(x) ||
                    (isfinite(x) && x >= 0 && isinteger(x)),
                df[!, c]
            )
        end
    end

    @testset "Processed data" begin

        # -------------------------------------------------
        # Surveillance
        # -------------------------------------------------

        @testset "National notifications" begin
            d = read_surv("tb_notifications_total")

            @test nrow(d) == 45
            @test issorted(d.year)
            check_unique(d, [:year])
            check_counts(d, [:cases])
        end

        @testset "Age-specific notifications" begin
            d = read_surv("tb_notifications_by_age")

            @test nrow(d) == 103
            check_unique(d, [:year, :age_group])
            check_counts(d, [:cases])

            allowed = Set([
                "0-4", "5-14", "0-14",
                "15-24", "25-34", "35-44",
                "45-54", "55-64", "65+"
            ])

            @test all(in(allowed), d.age_group)

            # WHO reported combined childhood data in 2024.
            d2024 = filter(:year => ==(2024), d)

            @test "0-14" in d2024.age_group
            @test !("0-4" in d2024.age_group)
            @test !("5-14" in d2024.age_group)
        end

        @testset "Recurrent notifications" begin
            d = read_surv("tb_recurrent_notifications")
            national = read_surv("tb_notifications_total")

            @test nrow(d) == 29
            check_unique(d, [:year])
            check_counts(d, [:cases])

            joined = innerjoin(
                rename(d, :cases => :recurrent),
                rename(national, :cases => :total);
                on=:year
            )

            @test nrow(joined) == nrow(d)
            @test all(
                joined.recurrent .<= joined.total
            )
        end

        @testset "Disease-site notifications" begin
            d = read_surv("tb_notifications_by_site")
            national = read_surv("tb_notifications_total")

            @test nrow(d) == 13
            check_unique(d, [:year])

            @test all(
                d.pulmonary_lab_confirmed .+
                    d.pulmonary_clinically_diagnosed .+
                    d.extrapulmonary .== d.total
            )

            joined = innerjoin(
                d,
                rename(
                    select(national, :year, :cases),
                    :cases => :national_total
                );
                on=:year
            )

            @test nrow(joined) == nrow(d)
            @test all(
                joined.total .== joined.national_total
            )
        end

        @testset "Treatment outcomes" begin
            all_tx = read_surv("tb_treatment_outcomes")
            main = read_surv("tb_treatment_outcomes_main")
            age_sex = read_surv(
                "tb_treatment_outcomes_age_sex"
            )

            @test nrow(all_tx) == 61
            @test nrow(main) == 13
            @test nrow(age_sex) == 9

            check_unique(all_tx, [:year, :who_cohort])
            check_unique(main, [:year])

            fields = [
                :successful, :died, :failed,
                :lost_to_followup, :not_evaluated
            ]

            check_counts(all_tx, [:cohort_size; fields])

            for r in eachrow(all_tx)
                values = [r[c] for c in fields]

                if all(!ismissing, values)
                    @test sum(values) == r.cohort_size
                end
            end

            @test all(
                r -> r.who_cohort ==
                    (r.year == 2024 ? "nrr_tx" : "newrel"),
                eachrow(main)
            )
        end

        @testset "Historical presentation" begin
            d = read_surv(
                "tb_historical_new_notifications_by_age_and_type"
            )

            sites = read_surv(
                "tb_new_notifications_by_site_age_2007_2012"
            )

            fractions = read_surv(
                "tb_pulmonary_fraction_by_age_2007_2012"
            )

            @test nrow(d) == 167
            @test nrow(sites) == 48
            @test nrow(fractions) == 8

            check_unique(
                d, [:year, :age_group, :who_category]
            )

            @test all(
                sites.pulmonary .+
                    sites.extrapulmonary .== sites.total
            )

            @test all(isapprox.(
                fractions.pulmonary_fraction,
                fractions.pulmonary ./ fractions.total;
                atol=1e-12
            ))
        end

        @testset "Documented WHO discrepancies" begin
            checks = CSV.read(
                joinpath(
                    survdir, "validation",
                    "notification_checks.csv"
                ),
                DataFrame
            )

            discrepancies = filter(
                :difference => x ->
                    !ismissing(x) && x != 0,
                checks
            )

            @test Dict(
                r.year => r.difference
                for r in eachrow(discrepancies)
            ) == Dict(2013 => 3, 2015 => 20)

            historical = CSV.read(
                joinpath(
                    survdir, "validation",
                    "historical_presentation_checks.csv"
                ),
                DataFrame
            )

            differences = filter(
                :difference => x ->
                    !ismissing(x) && x != 0,
                historical
            )

            @test Dict(
                (r.year, r.who_category) => r.difference
                for r in eachrow(differences)
            ) == Dict(
                (2006, "new_sn") => -14,
                (2006, "new_sp") => -4
            )
        end

        # -------------------------------------------------
        # WHO burden estimates
        # -------------------------------------------------

        @testset "WHO national burden" begin
            d = CSV.read(
                joinpath(
                    survdir, "estimates",
                    "tb_burden_national.csv"
                ),
                DataFrame
            )

            @test nrow(d) == 26
            @test d.year == collect(2000:2025)

            for (best, lo, hi) in (
                (:e_inc_num, :e_inc_num_lo, :e_inc_num_hi),
                (:e_mort_num, :e_mort_num_lo, :e_mort_num_hi)
            )
                for r in eachrow(d)
                    values = (r[best], r[lo], r[hi])

                    if all(!ismissing, values)
                        @test 0 <= r[lo] <= r[best] <= r[hi]
                    end
                end
            end
        end

        # -------------------------------------------------
        # Demography
        # -------------------------------------------------

        @testset "WPP2024 demography" begin
            population = read_demo("population")
            fertility = read_demo("fertility")
            mortality = read_demo("mortality")
            migration = read_demo("migration")

            @test nrow(population) == 15251
            @test nrow(fertility) == 6795
            @test nrow(mortality) == 15251
            @test nrow(migration) == 151

            for (d, ages) in (
                (population, 0:100),
                (fertility, 10:54),
                (mortality, 0:100)
            )
                check_unique(d, [:year, :age])

                for g in groupby(d, :year)
                    @test sort(g.age) == collect(ages)
                end
            end

            @test sort(unique(population.year)) ==
                collect(1950:2100)

            @test migration.year == collect(1950:2100)

            @test all(population.pop .>= 0)
            @test all(fertility.asfr .>= 0)
            @test all(mortality.mxB .>= 0)

            @test maximum(abs.(
                population.pop .-
                    population.popM .-
                    population.popF
            )) <= 1.01

            tfr = combine(
                groupby(fertility, :year),
                :asfr => sum => :reconstructed,
                :tfr => first => :reported
            )

            @test maximum(abs.(
                tfr.reconstructed .- tfr.reported
            )) < 1e-5

            p2023 = sum(
                population.pop[population.year .== 2023]
            )

            @test isapprox(p2023, 133527; atol=1)
        end

        # -------------------------------------------------
        # Contact matrices
        # -------------------------------------------------

        @testset "Prem contact matrices" begin
            original = CSV.read(
                joinpath(contactdir, "contact_matrix.csv"),
                DataFrame
            )

            corrected = CSV.read(
                joinpath(
                    contactdir,
                    "prem_reciprocal_2023_96.csv"
                ),
                DataFrame
            )

            @test nrow(original) == 96
            @test nrow(corrected) == 96

            C = Matrix{Float64}(original[:, 2:end])
            Cstar = Matrix{Float64}(corrected[:, 2:end])

            @test size(C) == (96, 96)
            @test size(Cstar) == (96, 96)
            @test all(isfinite, C)
            @test all(C .>= 0)
            @test all(Cstar .>= 0)

            C16 = Matrix{Float64}(readdlm(
                joinpath(
                    contactdir,
                    "prem_reciprocal_2023_16.csv"
                ), ','
            ))

            p = read_demo("population")
            p = filter(:year => ==(2023), p)

            N = zeros(16)

            for r in eachrow(p)
                group = min(div(r.age, 5) + 1, 16)
                N[group] += r.pop
            end

            flows = N .* C16

            relative_error =
                maximum(abs.(flows .- transpose(flows))) /
                    maximum(flows)

            @test relative_error < 1e-12

            # Correction must actually change mixing.
            @test norm(Cstar - C) > 0
        end

    end

    println("\nData validation complete.")
end
