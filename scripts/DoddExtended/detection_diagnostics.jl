
using TBKiribatiJulia
using DataFrames, CSV, Statistics

const DET = TBKiribatiJulia

problem_det = problem100
theta_det = fitted_models["F_nu_only"].parameters
model_det = problem_det.model

# Construct the actual parameters used by the ODE.
p_det = DET.make_parameters(model_det, theta_det)

results_dir_det = normpath(joinpath(
    @__DIR__, "..", "..", "results",
    "DoddExtended", "detection_diagnostics",
))
mkpath(results_dir_det)

age_breaks_det = problem_det.age_breaks

# --------------------------------------------------
# 1. Detection hazards and competing risks
# --------------------------------------------------

function detection_table(p, model, year, age_breaks)
    probability = DET.detection_probability(
        p.detection, year + 0.5
    )

    odds = probability / (1 - probability)

    categories = (
        "Smear positive",
        "Smear negative",
        "Extrapulmonary",
    )

    rows = NamedTuple[]

    for g in 1:(length(age_breaks)-1)
        ages = (age_breaks[g]+1):age_breaks[g+1]

        for k in 1:3
            for a in ages
                delta = p.detection_multiplier[a] *
                    odds * p.relative_detection[k]

                gamma = p.self_cure[a]
                mu = p.tb_mortality[a]

                exit_rate = delta + gamma + mu

                push!(rows, (
                    year=year,
                    age=a-1,
                    age_group=string(
                        age_breaks[g], "-",
                        age_breaks[g+1]-1
                    ),
                    presentation=categories[k],
                    onset_fraction=model.presentation[a, k],
                    detection_hazard=delta,
                    self_cure_hazard=gamma,
                    mortality_hazard=mu,
                    probability_detected=delta/exit_rate,
                    probability_self_cure=gamma/exit_rate,
                    probability_tb_death=mu/exit_rate,
                    mean_active_duration=1/exit_rate,
                ))
            end
        end
    end

    return DataFrame(rows)
end

hazards = vcat([
    detection_table(
        p_det, model_det, year, age_breaks_det
    )
    for year in (1980, 2000, 2013, 2024, 2025)
]...)

CSV.write(
    joinpath(results_dir_det, "detection_hazards.csv"),
    hazards,
)

# --------------------------------------------------
# 2. Age-specific presentation-weighted probabilities
# --------------------------------------------------

age_probabilities = combine(
    groupby(hazards, [:year, :age_group]),
    [:onset_fraction, :probability_detected] =>
        ((w, p) -> sum(w .* p) / (length(w)/3)) =>
            :probability_detected,
    [:onset_fraction, :probability_tb_death] =>
        ((w, p) -> sum(w .* p) / (length(w)/3)) =>
            :probability_tb_death,
    [:onset_fraction, :probability_self_cure] =>
        ((w, p) -> sum(w .* p) / (length(w)/3)) =>
            :probability_self_cure,
)

# The above weights sum to one separately for each age.
# Within each age group, we report an unweighted average
# across individual ages, not a population-weighted average.

CSV.write(
    joinpath(results_dir_det, "age_detection_probabilities.csv"),
    age_probabilities,
)

# --------------------------------------------------
# 3. Simulated annual outcomes
# --------------------------------------------------

out_det = predict(
    problem_det, theta_det;
    variables=(
        :incidence,
        :notifications,
        :mortality,
        :untreated_mortality,
        :treatment_mortality,
        :self_cure,
        :fast_progression,
        :slow_progression,
        :relapse,
        :pulmonary_notifications,
        :extrapulmonary_notifications,
        :smear_positive_notifications,
        :smear_negative_notifications,
    ),
)

total_det(variable, j) =
    sum(getproperty(out_det, variable)[:, j])

flow_rows = NamedTuple[]

for year in (1980, 2000, 2013, 2024, 2025)
    j = findfirst(==(year), out_det.years)

    incidence = total_det(:incidence, j)
    notifications = total_det(:notifications, j)
    deaths = total_det(:mortality, j)
    untreated = total_det(:untreated_mortality, j)
    cures = total_det(:self_cure, j)

    push!(flow_rows, (
        year=year,
        incidence=incidence,
        notifications=notifications,
        notifications_per_incidence=notifications/incidence,
        untreated_deaths=untreated,
        treatment_deaths=deaths-untreated,
        self_cure=cures,
        untreated_deaths_per_incidence=untreated/incidence,
        smear_positive=total_det(
            :smear_positive_notifications, j
        ),
        smear_negative=total_det(
            :smear_negative_notifications, j
        ),
        extrapulmonary=total_det(
            :extrapulmonary_notifications, j
        ),
        fast_progression=total_det(:fast_progression, j),
        slow_progression=total_det(:slow_progression, j),
        relapse=total_det(:relapse, j),
    ))
end

flows = DataFrame(flow_rows)

CSV.write(
    joinpath(results_dir_det, "annual_flows.csv"),
    flows,
)

# --------------------------------------------------
# 4. Aggregate competing risks, weighted by incidence
# --------------------------------------------------

# Within each surveillance age group, the presentation
# probabilities are constant by construction. Consequently,
# onset counts by age group can weight the probabilities.

j2024 = findfirst(==(2024), out_det.years)

risk_rows = NamedTuple[]

for g in 1:(length(age_breaks_det)-1)
    ages = (age_breaks_det[g]+1):age_breaks_det[g+1]
    label = string(
        age_breaks_det[g], "-",
        age_breaks_det[g+1]-1
    )

    onset = (
        out_det.fast_progression[g, j2024] +
            out_det.slow_progression[g, j2024] +
            out_det.relapse[g, j2024]
    )

    hg = filter(
        r -> r.year == 2024 && r.age_group == label,
        hazards,
    )

    push!(risk_rows, (
        age_group=label,
        incidence=onset,
        detection_probability=mean(
            filter(r -> r.age == first(ages)-1, hg).
            probability_detected
        ),
        actual_notifications=out_det.notifications[g, j2024],
    ))
end

# Replace simple category average with presentation-weighted
# probability from the age-specific table.
for g in 1:nrow(DataFrame(risk_rows))
    label = risk_rows[g].age_group
    p = only(filter(
        r -> r.year == 2024 && r.age_group == label,
        age_probabilities,
    ))

    risk_rows[g] = merge(
        risk_rows[g],
        (detection_probability=p.probability_detected,)
    )
end

risks = DataFrame(risk_rows)

weighted_detection = sum(
    risks.incidence .* risks.detection_probability
) / sum(risks.incidence)

CSV.write(
    joinpath(results_dir_det, "incidence_weighted_detection.csv"),
    risks,
)

# --------------------------------------------------
# 5. Results
# --------------------------------------------------

println("\nDETECTION PARAMETERS")
println("--------------------")
println("Detection trend: ", p_det.detection.trend)
println("Detection K: ", p_det.detection.K)
println("Maximum: ", p_det.detection.maximum)
println("Relative detection: ", p_det.relative_detection)
println("Untreated CFR: ", theta_det.untreated_cfr)
println("Adult duration: ", theta_det.duration)

println("\n2024 DETECTION HAZARDS")
println("----------------------")
display(filter(
    r -> r.year == 2024 && r.age in (2, 10, 30, 65),
    hazards,
))

println("\nAGE-SPECIFIC DETECTION")
println("----------------------")
display(filter(r -> r.year == 2024, age_probabilities))

println("\nINCIDENCE-WEIGHTED DETECTION PROBABILITY")
println("----------------------------------------")
println(weighted_detection)

println("\nANNUAL EPIDEMIOLOGICAL FLOWS")
println("---------------------------")
display(flows)

println("\nResults: ", results_dir_det)
