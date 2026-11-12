
using TBKiribatiJulia
using Test
using Statistics

# ============================================================
# 1. Configuration
# ============================================================

const NAGES = 96
const START_YEAR = 2023
const END_YEAR = 2026

const AGE_BREAKS = [
    0, 5, 15, 25, 35, 45, 55, 65, 96
]

const AGE_LABELS = [
    "0-4", "5-14", "15-24", "25-34",
    "35-44", "45-54", "55-64", "65+"
]

ages = collect(0:(NAGES - 1))
years = START_YEAR:(END_YEAR - 1)

# ============================================================
# 2. Synthetic demography
# ============================================================

# A declining population distribution by age.
population = 3000.0 .* exp.(-ages ./ 30)

# Build synthetic demographic data with a fixed population.
# The demographic operator still advances cohorts and
# reconciles each age group to its target population.

nyears = END_YEAR - START_YEAR + 1

demographic_data = DemographicData(
    repeat(
        reshape(population ./ 1000, :, 1),
        1, nyears
    ),
    zeros(86, nyears),       # Fertility
    zeros(NAGES, nyears),    # Background mortality
    zeros(nyears),          # Net migration
    START_YEAR,
    END_YEAR,
    10,                     # First fertility age
    0.485,                  # Female population fraction
)

demography = EquilibriumDemography(
    demographic_data,
    START_YEAR;
    nages = NAGES,
)

# ============================================================
# 3. Contact matrix and Dodd model
# ============================================================

# Illustrative age-assortative mixing.
contact = [
    exp(-abs(a - b) / 12)
    for a in ages, b in ages
]

model = DoddTBModel(contact)

layout = state_layout(model)

# Verify Dodd's contact normalisation.
@test sum(model.contact) / NAGES ≈ 1.0

println("Model age classes: ", layout.nages)
println("States per age: ", nstates(layout))
println("Total state dimension: ",
        layout.nages * nstates(layout))

# ============================================================
# 4. Epidemiological parameters
# ============================================================

detection = DoddDetection(
    K = 0.5,
    trend = 0.02,
    reference_year = START_YEAR,
    maximum = 0.9,
)

# Illustrative parameter values, not a calibrated estimate.
epsilon = [
    age < 5 ? 0.20 : 0.06
    for age in ages
]

duration = [
    age < 15 ? 0.5 : 3.0
    for age in ages
]

params = make_parameters(
    model;
    beta = 10.0,
    psi = 0.21,
    kappa = exp(0.62),
    epsilon = epsilon,
    nu = fill(exp(-6.89), NAGES),
    omega = fill(exp(-3.95), NAGES),
    duration = duration,
    fatality = fill(0.43, NAGES),
    relative_detection = ones(NAGES),
    treatment_rate = 2.0,
    treatment_cfr = 0.03,
    detection = detection,
)

# ============================================================
# 5. Initial conditions
# ============================================================

Lambda0 = 0.03

u0 = initial_state(
    model,
    population,
    Lambda0,
    params;
    year = START_YEAR,
)

# Initial population conservation by age.
for a in 1:NAGES
    total = sum(
        u0[state_index(layout, a, c)]
        for c in 1:layout.nepi
    )

    @test total ≈ population[a] rtol=1e-10
end

println("Initial population: ", sum(population))

# ============================================================
# 6. Run simulation
# ============================================================

sol = simulate(
    model,
    params,
    demography;
    u0 = u0,
    tspan = (START_YEAR, END_YEAR),
)

println("\nSimulation completed")
println("Return code: ", sol.retcode)
println("Saved states: ", length(sol.t))

# ============================================================
# 7. Check annual demographic boundaries
# ============================================================

println("\nSaved solution times:")
println(sol.t)

# Internal demographic updates should have two states.
for year in (START_YEAR + 1):(END_YEAR - 1)
    indices = findall(t -> isapprox(t, year; atol=1e-8),
                      sol.t)

    @test length(indices) >= 2

    before = sol.u[first(indices)]
    after = sol.u[last(indices)]

    # Cumulative event counters must be unaffected
    # by the demographic operator.
    for a in 1:NAGES
        for c in (layout.nepi + 1):nstates(layout)
            i = state_index(layout, a, c)
            @test before[i] ≈ after[i] atol=1e-8
        end
    end
end

# ============================================================
# 8. Extract annual outputs
# ============================================================

out = model_outputs(
    model,
    sol;
    years = years,
    age_breaks = AGE_BREAKS,
)

println("\nAnnual expected TB outcomes")
println("Year      Incidence    Notifications    Mortality")

for (j, year) in enumerate(out.years)
    incidence = sum(out.incidence[:, j])
    notifications = sum(out.notifications[:, j])
    mortality = sum(out.mortality[:, j])

    println(
        year, "      ",
        round(incidence; digits=1), "         ",
        round(notifications; digits=1), "          ",
        round(mortality; digits=1)
    )
end

# ============================================================
# 9. Validate output identities
# ============================================================

eta = model.pulmonary_fraction

# Dodd output components are pulmonary-only.
@test out.incidence ≈ (
    out.fast_progression .+
    out.slow_progression .+
    out.relapse
) ./ eta

# Compare notification output with the raw
# treatment-initiation counter.
treatment_counts = annual_events(
    sol,
    model,
    12;  # CumTreatmentInitiation
    years = years,
)

aggregated_treatment = aggregate_age_groups(
    treatment_counts,
    AGE_BREAKS,
)

@test out.notifications ≈ aggregated_treatment ./ eta

# All extracted counts should be finite and nonnegative.
for name in (
    :incidence,
    :notifications,
    :mortality,
    :primary_infections,
    :reinfections,
)
    values = getproperty(out, name)
    @test all(isfinite, values)
    @test all(x -> x >= 0, values)
end

# ============================================================
# 10. Age-specific results
# ============================================================

println("\nAge-specific outcomes in ", first(years))
println("Age group    Incidence    Notifications")

for (g, label) in enumerate(AGE_LABELS)
    println(
        rpad(label, 12),
        round(out.incidence[g, 1]; digits=1),
        "          ",
        round(out.notifications[g, 1]; digits=1)
    )
end

println("\nAll checks passed.")
