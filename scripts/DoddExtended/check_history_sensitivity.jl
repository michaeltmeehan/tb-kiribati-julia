
using Test
using Statistics: median
using TBKiribatiJulia
using SciMLBase: successful_retcode

const DE = TBKiribatiJulia.DoddExtendedStates

include(
    joinpath(@__DIR__, "..", "common", "kiribati_setup.jl")
)

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

setup = setup_dodd_analysis(
    start_year=1970,
    end_year=2026,
    output_years=2013:2025,
)

data_dir = normpath(
    joinpath(@__DIR__, "..", "..", "data")
)

presentation = load_kiribati_presentation(
    data_dir;
    years=2007:2012,
).presentation

model = DoddExtendedTBModel(
    setup.model.contact,
    presentation;
    reinfection=true,
)

theta = merge(
    map(median, priors(model)),
    extended_fixed_parameters(),
)

params = make_parameters(model, theta)

u0 = initial_state(
    model,
    params,
    setup.demography;
    year=1970,
)


# ------------------------------------------------------------
# Change history allocation without changing total states
# ------------------------------------------------------------

function redistribute_history(u, model, fraction)
    0 <= fraction <= 1 ||
        throw(ArgumentError("Invalid fraction"))

    v = copy(u)
    layout = state_layout(model)
    stride = nstates(layout)

    pairs = (
        (DE.E0, DE.E1),
        (DE.L0, DE.L1),
        (DE.I0SP, DE.I1SP),
        (DE.I0SN, DE.I1SN),
        (DE.I0EP, DE.I1EP),
    )

    for a in 1:layout.nages
        b = (a - 1) * stride

        for (c0, c1) in pairs
            transferred = fraction * v[b+c0]

            v[b+c0] -= transferred
            v[b+c1] += transferred
        end
    end

    return v
end


# ------------------------------------------------------------
# Simulate alternative initial histories
# ------------------------------------------------------------

fractions = (0.0, 0.5, 1.0)
outputs = Dict{Float64,Any}()

for fraction in fractions

    println("\nInitial history fraction: ", fraction)

    initial = redistribute_history(
        u0, model, fraction
    )

    sol = simulate(
        model,
        params,
        setup.demography;
        u0=initial,
        tspan=(1970.0, 2026.0),
    )

    @test successful_retcode(sol)

    out = model_outputs(
        model,
        sol;
        years=1996:2025,
        age_breaks=setup.age_breaks,
    )

    @test isapprox(
        out.notifications,
        out.new_notifications .+
            out.recurrent_notifications;
        atol=1e-5,
        rtol=1e-7,
    )

    outputs[fraction] = out
end


# ------------------------------------------------------------
# Quantify differences
# ------------------------------------------------------------

national(x) = vec(sum(x; dims=1))

function maximum_relative_difference(
    baseline, alternative;
    years, start_year,
)
    idx = findall(>=(start_year), years)

    isempty(idx) &&
        throw(ArgumentError("No years selected"))

    x = baseline[idx]
    y = alternative[idx]

    return maximum(
        abs.(y .- x) ./ max.(1.0, abs.(x))
    )
end


baseline = outputs[0.0]

for fraction in (0.5, 1.0)
    alternative = outputs[fraction]

    println("\nHistory allocation: ", fraction)

    for start_year in (1996, 2013)
        total_difference = maximum_relative_difference(
            national(baseline.notifications),
            national(alternative.notifications);
            years=baseline.years,
            start_year=start_year,
        )

        recurrence_difference = maximum_relative_difference(
            national(baseline.recurrent_notifications),
            national(alternative.recurrent_notifications);
            years=baseline.years,
            start_year=start_year,
        )

        println(
            "Since $start_year: ",
            "total difference = ",
            round(100 * total_difference; digits=3),
            "%, recurrence difference = ",
            round(100 * recurrence_difference; digits=3),
            "%",
        )

        # History labels should not alter aggregate dynamics.
        @test total_difference < 1e-4

        # Recurrent notifications may remain history-sensitive.
        if start_year == 2013 &&
            recurrence_difference > 0.05
            @warn "Recent recurrence predictions remain sensitive to initial treatment history" fraction recurrence_difference
        end
    end
end
