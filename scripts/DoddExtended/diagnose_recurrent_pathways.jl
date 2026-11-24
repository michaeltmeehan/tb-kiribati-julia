
using TBKiribatiJulia
using DataFrames, CSV

# Requires calibration_component_comparison.jl
# to have been executed in the current Julia session.

let
    TB = TBKiribatiJulia
    DE = TB.DoddExtendedStates

    problem = problems[1]
    theta = fits[1].parameters
    model = problem.model

    p = TB.make_parameters(
        model,
        TB.model_draw(problem, theta),
    )

    sol = TB.simulate_from_draw(
        problem.simulation,
        TB.model_draw(problem, theta),
    )

    stride = TB.nstates(model.layout)
    nages = model.layout.nages
    nepi = model.layout.nepi

    # Use midpoint integration, avoiding demographic
    # discontinuities at integer years.
    nsteps = 24
    dt = 1.0 / nsteps

    function rates(t)
        u = sol(t)

        infectious = zeros(nages)
        population = zeros(nages)

        for a in 1:nages
            b = (a - 1) * stride

            population[a] = sum(
                u[b+c] for c in 1:nepi
            )

            infectious[a] =
                model.infectiousness[a] * (
                    p.relative_infectiousness[1] *
                            (u[b+DE.I0SP] + u[b+DE.I1SP]) +
                        p.relative_infectiousness[2] *
                            (u[b+DE.I0SN] + u[b+DE.I1SN]) +
                        p.relative_infectiousness[3] *
                            (u[b+DE.I0EP] + u[b+DE.I1EP])
                ) / max(population[a], eps())
        end

        foi = p.beta .* (model.contact * infectious)

        d = TB.detection_probability(p.detection, t)
        odds = d / (1 - d)

        totals = zeros(9)

        for a in 1:nages
            b = (a - 1) * stride
            lambda = foi[a]

            E1 = u[b+DE.E1]
            L1 = u[b+DE.L1]
            R = u[b+DE.R]

            fast = p.epsilon[a] * E1
            slow = p.nu[a] * L1
            relapse = p.omega[a] * R

            reinf_L1 = p.psi * lambda * L1
            reinf_R = p.psi * lambda * R

            recurrent_notifications = 0.0
            new_notifications = 0.0

            for k in 1:3
                delta = odds *
                    p.detection_multiplier[a] *
                    p.relative_detection[k]

                recurrent_notifications +=
                    delta * u[b+DE.I1SP+k-1]

                new_notifications +=
                    delta * u[b+DE.I0SP+k-1]
            end

            totals .+= [
                fast,
                slow,
                relapse,
                reinf_L1,
                reinf_R,
                recurrent_notifications,
                new_notifications,
                L1,
                R,
            ]
        end

        return totals
    end

    years = 1996:2024
    records = NamedTuple[]

    for year in years
        annual = zeros(9)

        for step in 1:nsteps
            t = year + (step - 0.5) * dt
            annual .+= dt .* rates(t)
        end

        fast, slow, relapse, reinf_L1,
        reinf_R, recurrent, new,
        mean_L1, mean_R = annual

        onset = fast + slow + relapse

        push!(records, (
            year=year,
            recurrent_fast_onset=fast,
            recurrent_slow_onset=slow,
            relapse_onset=relapse,
            total_recurrent_onset=onset,
            fast_onset_fraction=fast/onset,
            slow_onset_fraction=slow/onset,
            relapse_onset_fraction=relapse/onset,
            reinfections_L1=reinf_L1,
            reinfections_R=reinf_R,
            recurrent_notifications=recurrent,
            new_notifications=new,
            recurrent_fraction=recurrent/(new+recurrent),
            mean_L1=mean_L1,
            mean_R=mean_R,
        ))
    end

    result = DataFrame(records)

    # Compare the numerical integration with the
    # exact cumulative-counter outputs.
    out = TB.predict(
        problem,
        theta;
        variables=(
            :new_notifications,
            :recurrent_notifications,
        ),
    )

    counter_errors = Float64[]

    for row in eachrow(result)
        j = findfirst(==(row.year), out.years)

        if !isnothing(j)
            push!(
                counter_errors,
                abs(
                    row.recurrent_notifications -
                        sum(out.recurrent_notifications[:, j])
                ),
            )
        end
    end

    println("Maximum midpoint integration error: ",
        maximum(counter_errors))

    println("\nRECURRENT DISEASE PATHWAYS")
    display(filter(
        r -> r.year in (1996, 2005, 2015, 2023, 2024),
        select(
            result,
            :year,
            :recurrent_fast_onset,
            :recurrent_slow_onset,
            :relapse_onset,
            :total_recurrent_onset,
            :recurrent_notifications,
            :recurrent_fraction,
        ),
    ))

    println("\nREINFECTION OF PREVIOUSLY TREATED PEOPLE")
    display(filter(
        r -> r.year in (1996, 2005, 2015, 2023, 2024),
        select(
            result,
            :year,
            :reinfections_L1,
            :reinfections_R,
            :mean_L1,
            :mean_R,
        ),
    ))

    dir = joinpath(
        "results", "DoddExtended",
        "calibration_component_comparison",
    )
    mkpath(dir)

    CSV.write(
        joinpath(dir, "recurrent_pathways_A.csv"),
        result,
    )
end
