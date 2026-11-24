using TBKiribatiJulia
using Statistics
using DataFrames
using Test
using DelimitedFiles

# ============================================================
# Configuration
# ============================================================

ROOT = normpath(joinpath(@__DIR__, "..", ".."))
DATA_DIR = joinpath(ROOT, "data")

NAGES = 96
START_YEAR = 2025
END_YEAR = 2050
YEARS = collect(START_YEAR:END_YEAR)
REPORT_YEARS = (2025, 2030, 2040, 2050)

# ============================================================
# Locate data files
# ============================================================

demography_dirs = String[]
presentation_dirs = String[]
contact_paths = String[]

for (dir, _, files) in walkdir(DATA_DIR)

    if all(f -> f in files,
        ("population", "fertility", "mortality", "migration"))
        push!(demography_dirs, dir)
    end

    if "tb_historical_new_notifications_by_age_and_type.csv" in files
        push!(presentation_dirs, dir)
    end

    for file in files
        if occursin("contact", lowercase(file))
            path = joinpath(dir, file)

            try
                load_contact_matrix(path; nages=NAGES)
                push!(contact_paths, path)
            catch
                # Ignore incompatible contact files.
            end
        end
    end
end

@assert length(demography_dirs) == 1 demography_dirs
@assert length(presentation_dirs) == 1 presentation_dirs
@assert length(contact_paths) == 1 contact_paths

DEMOGRAPHY_DIR = only(demography_dirs)
PRESENTATION_DIR = only(presentation_dirs)
CONTACT_PATH = only(contact_paths)

println("Demography:   ", DEMOGRAPHY_DIR)
println("Presentation: ", PRESENTATION_DIR)
println("Contacts:     ", CONTACT_PATH)

# ============================================================
# Data and model
# ============================================================

data = load_demographic_data(DEMOGRAPHY_DIR)

contact = load_contact_matrix(
    CONTACT_PATH;
    nages=NAGES,
)

presentation_data = load_kiribati_presentation(
    PRESENTATION_DIR;
    nages=NAGES,
)

model = DoddExtendedTBModel(
    contact,
    presentation_data.presentation,
)

layout = state_layout(model)

# ============================================================
# Epidemiological parameters
# ============================================================

draw = map(median, priors(model))
params = make_parameters(model, draw)

# ============================================================
# Common initial state
# ============================================================

demography = ProjectedDemography(data; nages=NAGES)

u0 = initial_state(
    model,
    params,
    demography;
    year=START_YEAR,
)

# ============================================================
# Run projections
# ============================================================

results = project_scenarios(
    model,
    params,
    data;
    u0=u0,
    start_year=START_YEAR,
    end_year=END_YEAR,
    variables=(
        :population,
        :incidence,
        :notifications,
        :mortality,
        :latent_prevalence,
        :active_prevalence,
    ),
)

# ============================================================
# Projection summaries and comparisons
# ============================================================

national = summarise_projections(results, YEARS)

comparison = compare_projections(national)

# ============================================================
# Validate deterministic projections
# ============================================================

@testset "Deterministic projections" begin

    counter_indices = [
        state_index(layout, a, c)
        for a in 1:layout.nages
        for c in (layout.nepi+1):nstates(layout)
    ]

    for (scenario, result) in pairs(results)

        @testset "$scenario" begin

            o = result.outputs
            sol = result.solution

            @test first(sol.u) ≈ u0

            # Validate annual population targets.

            for (j, year) in enumerate(YEARS)

                target = get_population(
                    result.demography,
                    year,
                )

                @test isapprox(
                    o.population[:, j],
                    target;
                    rtol=1e-6,
                    atol=1e-6,
                )
            end

            # Validate epidemiological outputs.

            for variable in (
                :incidence,
                :notifications,
                :mortality,
                :latent_prevalence,
                :active_prevalence,
            )
                values = getproperty(o, variable)

                @test all(isfinite, values)
                @test minimum(values) >= -1e-8
            end

            # Demographic callbacks must preserve counters.

            for i in 2:length(sol.t)
                if sol.t[i] == sol.t[i-1]

                    before = sol.u[i-1][counter_indices]
                    after = sol.u[i][counter_indices]

                    @test before ≈ after
                end
            end
        end
    end

    N_fixed = results.fixed.outputs.population
    N_growth = results.growth_only.outputs.population
    N_projected = results.projected.outputs.population

    # Fixed population remains constant.

    @test N_fixed ≈ repeat(
        N_fixed[:, 1:1], 1, length(YEARS)
    )

    # Growth-only and projected population totals agree.

    @test sum(N_growth; dims=1) ≈
        sum(N_projected; dims=1)

    # Growth-only age distribution remains constant.

    proportions = N_growth ./ sum(N_growth; dims=1)

    @test proportions ≈ repeat(
        proportions[:, 1:1], 1, length(YEARS)
    )

    # Growth-only proportional-scaling invariance.

    for variable in (:incidence, :notifications, :mortality)

        fixed_rate =
            getproperty(results.fixed.outputs, variable) ./
                N_fixed

        growth_rate =
            getproperty(results.growth_only.outputs, variable) ./
                N_growth

        @test isapprox(
            fixed_rate,
            growth_rate;
            rtol=1e-4,
        )
    end
end

# ============================================================
# Validate projection decomposition
# ============================================================

@testset "Projection decomposition" begin

    for variable in unique(comparison.variable)

        subset = filter(
            r -> r.variable == variable,
            comparison,
        )

        growth = filter(
            r -> r.effect == :population_growth,
            subset,
        )

        composition = filter(
            r -> r.effect == :demographic_composition,
            subset,
        )

        combined = filter(
            r -> r.effect == :combined_demography,
            subset,
        )

        # Additive decomposition.

        @test combined.difference ≈
            growth.difference .+ composition.difference

        # Multiplicative decomposition.

        valid = .!ismissing.(combined.ratio) .&
            .!ismissing.(growth.ratio) .&
            .!ismissing.(composition.ratio)

        @test combined.ratio[valid] ≈
            growth.ratio[valid] .* composition.ratio[valid]
    end

    # Growth-only incidence rate invariance.

    growth_rates = filter(
        r -> r.effect == :population_growth &&
            r.variable == :incidence_rate,
        comparison,
    )

    @test all(
        isapprox.(growth_rates.ratio, 1.0; rtol=1e-4)
    )
end

# ============================================================
# Display results
# ============================================================

println("\nNATIONAL PROJECTIONS")

display(filter(
    r -> r.year in REPORT_YEARS,
    national,
))

println("\nINCIDENCE COMPARISONS")

display(select(
    filter(
        r -> r.year in REPORT_YEARS &&
            r.variable == :incidence,
        comparison,
    ),
    :year,
    :effect,
    :numerator,
    :denominator,
    :difference,
    :ratio,
    :percentage_change,
))

# ============================================================
# Visualisation
# ============================================================

using CairoMakie

fig_demography = plot_projections(
    national;
    variables=(
        :population,
        :proportion_65_plus,
    ),
)

fig_burden = plot_projections(
    national;
    variables=(
        :incidence,
        :incidence_rate,
        :notifications,
        :notification_rate,
        :mortality,
        :mortality_rate,
    ),
)

fig_effects = plot_projection_effects(
    comparison;
    variables=(
        :incidence,
        :incidence_rate,
        :mortality,
        :mortality_rate,
    ),
)

display(fig_demography)
display(fig_burden)
display(fig_effects)

# ============================================================
# Save figures
# ============================================================

OUTPUT_DIR = joinpath(
    ROOT, "results", "projections", "deterministic"
)

mkpath(OUTPUT_DIR)

save(joinpath(OUTPUT_DIR, "demography.png"), fig_demography)
save(joinpath(OUTPUT_DIR, "tb_burden.png"), fig_burden)
save(joinpath(OUTPUT_DIR, "demographic_effects.png"), fig_effects)