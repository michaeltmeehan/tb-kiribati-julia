# Independent natural-history prior-predictive diagnostic for DoddExtended.
# Run from the repository root:
#   julia --project=. scripts/DoddExtended/natural_history_prior_predictive.jl
# No calibration results, surveillance likelihood or ODE simulations are used.

using TBKiribatiJulia
using Distributions
using Random
using Statistics
using DataFrames
using CSV
using CairoMakie

const TB = TBKiribatiJulia
const N_DRAWS = 10_000
const SEED = 20261123
const OUTDIR = joinpath("results", "DoddExtended", "natural_history_priors")
mkpath(OUTDIR)

# Historical Kiribati notification presentation data.
# These are NOT disease-onset probabilities.
const NAGES = 96

const SURVEILLANCE_DIR = joinpath(
    "data", "processed", "surveillance"
)

presentation_data = TB.load_kiribati_presentation(
    SURVEILLANCE_DIR;
    years=2007:2012,
    nages=NAGES,
)

Q_observed = presentation_data.presentation

# The contact matrix is irrelevant here because
# we do not simulate transmission.
C = ones(NAGES, NAGES)

model = TB.DoddExtendedTBModel(
    C, Q_observed
)

prior = TB.priors(model)
rng = MersenneTwister(SEED)

println("Loaded Kiribati presentation data (2007–2012)")
println("Notification counts by age group:")
println(presentation_data.counts)

# Lifetime probability of fast progression *before stabilisation* for a newly
# infected person starting at an exact integer age. Includes subsequent age
# changes in epsilon and kappa; excludes background mortality and other risks.
# The final age-specific rates are held constant after the last modelled age.
function effective_fast_probability(epsilon, kappa, start_age::Int)
    0 <= start_age < length(epsilon) || throw(BoundsError(epsilon, start_age + 1))
    survival = 1.0
    progressed = 0.0
    for a in (start_age+1):length(epsilon)
        ef = epsilon[a]
        h = ef + kappa[a]
        if h > 0
            exits = -expm1(-h)    # one-year interval
            progressed += survival * exits * ef / h
            survival *= exp(-h)
        end
    end
    ef = epsilon[end]
    h = ef + kappa[end]
    h > 0 && (progressed += survival * ef / h)
    return progressed
end

"""
Cumulative probability of active TB following infection.

Starts entirely in E at age_at_infection and follows
E -> disease (epsilon), E -> L (kappa),
and L -> disease (nu).

Includes ageing across annual age-specific hazards.
Excludes reinfection, competing mortality and recovery.
"""
function disease_risk_after_infection(
    p,
    age_at_infection::Int,
    followup::Real,
)
    followup >= 0 ||
        throw(ArgumentError("Negative follow-up"))

    E = 1.0
    L = 0.0
    fast = 0.0
    slow = 0.0
    elapsed = 0.0

    while elapsed < followup
        age = age_at_infection + floor(Int, elapsed)
        a = min(age + 1, length(p.epsilon))

        dt = min(1.0, followup - elapsed)

        ε = p.epsilon[a]
        κ = p.kappa[a]
        ν = p.nu[a]

        h = ε + κ

        Enew = E * exp(-h * dt)

        Lnew = if isapprox(h, ν; atol=1e-12)
            exp(-ν * dt) * (L + κ * E * dt)
        else
            L * exp(-ν * dt) +
                κ * E *
                (exp(-ν * dt) - exp(-h * dt)) /
                (h - ν)
        end

        fast_increment =
            h > 0 ? ε * E * (-expm1(-h * dt)) / h : 0.0

        slow_increment =
            E + L - Enew - Lnew - fast_increment

        fast += fast_increment
        slow += slow_increment

        E = Enew
        L = Lnew
        elapsed += dt
    end

    return (
        fast=fast,
        slow=slow,
        total=fast + slow,
        early_remaining=E,
        latent_remaining=L,
    )
end

# Long-format raw draws, compact summary and plots.
metric_col = String[]
age_col = Int[]
presentation_col = String[]
value_col = Float64[]

function record!(metric, age, presentation, value)
    push!(metric_col, String(metric))
    push!(age_col, Int(age))
    push!(presentation_col, String(presentation))
    push!(value_col, Float64(value))
end

ages = (0, 1, 3, 7, 12, 15, 25, 45, 75)
sites = ("SP", "SN", "EP")
for _ in 1:N_DRAWS
    draw = (; (k => rand(rng, distribution) for (k, distribution) in pairs(prior))...)
    p = TB.make_parameters(model, draw)
    ascertainment = TB.presentation_ascertainment(p, 2023.0)

    # Invert the historical notification proportions
    # using presentation-specific ascertainment.
    P_onset = TB.inferred_onset_presentation(
        Q_observed,
        ascertainment,
    )

    # Verify the forward/inverse calculation.
    Q_reconstructed = TB.expected_notification_presentation(
        P_onset,
        ascertainment,
    )

    @assert isapprox(
        Q_reconstructed, Q_observed;
        atol=1e-10,
    )

    A = TB.presentation_ascertainment(p, 2023.0)

    # Historical Kiribati notification distribution.
    Q = model.presentation

    # Inferred underlying disease-onset distribution.
    P_onset = TB.inferred_onset_presentation(Q, A)

    # Examine selected ages.
    for age in (0, 3, 7, 12, 15, 25)
        a = age + 1

        for (k, site) in enumerate(sites)
            record!(
                "presentation_notified",
                age, site, Q_observed[a, k]
            )

            record!(
                "presentation_onset_corrected",
                age, site, P_onset[a, k]
            )
        end

        for k in 1:3
            # Store these using the existing record! interface,
            # or the equivalent output collector in the script.
            record!(
                "presentation_notified",
                age, ("SP", "SN", "EP")[k], Q[a, k]
            )

            record!(
                "presentation_onset_corrected",
                age, ("SP", "SN", "EP")[k], P_onset[a, k]
            )
        end
    end

    for age in ages
        a = age + 1
        local_risk = p.epsilon[a] / (p.epsilon[a] + p.kappa[a])
        record!("fast_risk_local", age, "all", local_risk)
        record!("fast_risk_effective", age, "all",
            effective_fast_probability(p.epsilon, p.kappa, age))
        record!("slow_progression_hazard", age, "all", p.nu[a])

        # A mixture of natural-history CFRs is only a scenario under the
        # illustrative onset presentation proportions P, NOT a Kiribati CFR.
        cfr = p.tb_mortality[a, :] ./
            (p.tb_mortality[a, :] .+ p.self_cure[a, :])
        record!("cfr_mixture_illustrative", age, "all",
            sum(p.presentation_fast[a, :] .* cfr))
        record!(
            "cfr_mixture_corrected",
            age, "all",
            sum(P_onset[a, :] .* cfr),
        )
        for (k, site) in enumerate(sites)
            gamma = p.self_cure[a, k]
            mu = p.tb_mortality[a, k]
            record!("mortality_hazard", age, site, mu)
            record!("self_cure_hazard", age, site, gamma)
            record!("natural_duration", age, site, 1 / (gamma + mu))
            record!("untreated_cfr", age, site, mu / (gamma + mu))
            record!("eventual_ascertainment", age, site, ascertainment[a, k])
        end
        for horizon in (1, 2, 5, 10)
            risk = disease_risk_after_infection(
                p, age, horizon
            )

            record!(
                "disease_risk_$(horizon)y",
                age, "all", risk.total
            )

            record!(
                "fast_risk_$(horizon)y",
                age, "all", risk.fast
            )

            record!(
                "slow_risk_$(horizon)y",
                age, "all", risk.slow
            )
        end
    end
    record!("relapse_hazard", -1, "all", p.omega[1])
    record!("reinfection_multiplier", -1, "all", p.psi)
    record!("sn_relative_infectiousness", -1, "all", p.relative_infectiousness[2])
    record!("beta_contact_normalisation_dependent", -1, "all", p.beta)
end

raw = DataFrame(metric=metric_col, age=age_col,
    presentation=presentation_col, value=value_col)
CSV.write(joinpath(OUTDIR, "draws.csv"), raw)

summary = combine(groupby(raw, [:metric, :age, :presentation]),
    :value => median => :median,
    :value => (v -> quantile(v, 0.025)) => :q025,
    :value => (v -> quantile(v, 0.975)) => :q975,
    :value => mean => :mean,
)
sort!(summary, [:metric, :age, :presentation])
CSV.write(joinpath(OUTDIR, "summary.csv"), summary)

# Compare historical notified proportions with
# inferred disease-onset proportions.
fig_p = Figure(size=(1200, 900))

for (k, site) in enumerate(sites)
    ax = Axis(
        fig_p[k, 1],
        xlabel="Age (years)",
        ylabel="Proportion",
        title="$site presentation",
        limits=(nothing, (0, 1)),
    )

    notified = sort(
        filter(
            r -> r.metric == "presentation_notified" &&
                r.presentation == site,
            summary,
        ),
        :age,
    )

    onset = sort(
        filter(
            r -> r.metric == "presentation_onset_corrected" &&
                r.presentation == site,
            summary,
        ),
        :age,
    )

    xs = Float64.(notified.age)

    lines!(
        ax, xs, notified.median;
        color=:black,
        linewidth=2,
        label="Notified",
    )

    band!(
        ax, xs, onset.q025, onset.q975;
        color=(:steelblue, 0.2),
    )

    lines!(
        ax, xs, onset.median;
        color=:steelblue,
        linewidth=2,
        label="Corrected onset",
    )

    axislegend(ax, position=:rt)
end

save(
    joinpath(OUTDIR, "presentation_correction.png"),
    fig_p,
)

function panel_data(metric, presentation)
    sort(filter(row -> row.metric == metric &&
        row.presentation == presentation, summary), :age)
end

# Visualise hazard / duration / CFR distributions across age groups.
fig = Figure(size=(1300, 950))
plot_specs = (
    ("untreated_cfr", "Untreated CFR", "probability"),
    ("natural_duration", "Mean untreated duration", "years"),
    ("eventual_ascertainment", "Eventual detection before competing exit", "probability"),
)
site_colors = (:steelblue, :darkorange, :forestgreen)
for (i, (metric, label, ylabel)) in enumerate(plot_specs)
    ax = Axis(fig[i, 1], xlabel="Age at disease onset (years)",
        ylabel=ylabel, title=label)
    for (k, site) in enumerate(sites)
        d = panel_data(metric, site)
        xs = Float64.(d.age)
        ys = d.median
        col = site_colors[k]
        band!(ax, xs, d.q025, d.q975, color=(col, 0.14))
        lines!(ax, xs, ys, color=col, linewidth=2, label=site)
        scatter!(ax, xs, ys, color=col, markersize=7)
    end
    axislegend(ax, position=:rt)
end
save(joinpath(OUTDIR, "natural_history_by_age.png"), fig)

fig2 = Figure(size=(1100, 560))
ax = Axis(fig2[1, 1], xlabel="Age at infection (years)",
    ylabel="Probability of fast progression",
    title="Local competing-hazard probability vs ageing-adjusted probability")
for (metric, label, color) in (
    ("fast_risk_local", "Local risk parameter", :steelblue),
    ("fast_risk_effective", "Ageing-adjusted risk", :darkorange),
)
    d = panel_data(metric, "all")
    xs = Float64.(d.age)
    band!(ax, xs, d.q025, d.q975, color=(color, 0.18))
    lines!(ax, xs, d.median, color=color, linewidth=2, label=label)
    scatter!(ax, xs, d.median, color=color, markersize=8)
end
axislegend(ax, position=:rt)
save(joinpath(OUTDIR, "fast_progression.png"), fig2)

println("\nNatural-history prior predictive completed: $N_DRAWS draws")
println("Outputs: $(abspath(OUTDIR))")
println("\nSelected benchmarks (medians and 95% prior intervals):")
for (metric, age, site) in (
    ("fast_risk_effective", 0, "all"),
    ("fast_risk_effective", 7, "all"),
    ("fast_risk_effective", 15, "all"),
    ("cfr_mixture_illustrative", 3, "all"),
    ("cfr_mixture_illustrative", 7, "all"),
    ("natural_duration", 25, "SP"),
    ("natural_duration", 25, "SN"),
    ("eventual_ascertainment", 25, "SP"),
    ("eventual_ascertainment", 25, "SN"),
)
    d = filter(row -> row.metric == metric && row.age == age &&
        row.presentation == site, summary)
    r = only(eachrow(d))
    println(rpad("$metric, age $age, $site", 51),
        round(r.median; sigdigits=3), " [",
        round(r.q025; sigdigits=3), ", ",
        round(r.q975; sigdigits=3), "]")
end
println("\nChildhood untreated CFR literature benchmarks:")
println("  ages 0–4: 0.436 (95% CI 0.368–0.506)")
println("  ages 5–14: 0.149 (95% CI 0.115–0.191)")
println("Illustrative model mixtures above use ages 3 and 7, not an")
println("age/incidence-weighted national childhood case distribution.")
