
using Test
using LinearAlgebra
using Statistics: median
using DifferentialEquations: ODEProblem, solve, Tsit5
using SciMLBase: successful_retcode
using TBKiribatiJulia

const TB = TBKiribatiJulia
const DE = TB.DoddExtendedStates

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function synthetic_initial_state(model)
    layout = state_layout(model)
    u = zeros(layout.nages * nstates(layout))

    for a in 1:layout.nages
        b = (a - 1) * nstates(layout)

        u[b+TB.S] = 850.0
        u[b+TB.E] = 30.0
        u[b+TB.L] = 90.0
        u[b+TB.I] = 10.0
        u[b+TB.T] = 8.0
        u[b+TB.R] = 12.0
    end

    return u
end


function rhs(model, params, u; t=2023.0)
    ctx = TB.SimulationContext(
        model,
        params,
        nothing,
        TB.TBWorkspace(state_layout(model).nages),
    )

    du = zeros(length(u))
    TB.tb_rhs!(du, u, ctx, t)
    return du
end


function solve_without_demography(
    model, params, u0;
    tspan=(2020.0, 2025.0),
)
    ctx = TB.SimulationContext(
        model,
        params,
        nothing,
        TB.TBWorkspace(state_layout(model).nages),
    )

    prob = ODEProblem(
        TB.tb_rhs!,
        copy(u0),
        tspan,
        ctx,
    )

    return solve(
        prob,
        Tsit5();
        saveat=tspan[1]:1.0:tspan[2],
        abstol=1e-10,
        reltol=1e-9,
    )
end


# Return only the six original epidemiological states.
function original_epi(u, model)
    layout = state_layout(model)
    stride = nstates(layout)

    result = zeros(6 * layout.nages)

    for a in 1:layout.nages
        old = (a - 1) * stride
        new = (a - 1) * 6

        result[(new+1):(new+6)] .=
            @view u[(old+1):(old+6)]
    end

    return result
end


# Aggregate history/presentation states to Dodd's six states.
function collapse_extended(u, model)
    layout = state_layout(model)
    stride = nstates(layout)

    result = zeros(6 * layout.nages)

    for a in 1:layout.nages
        b = (a - 1) * stride
        j = (a - 1) * 6

        result[j+1] = u[b+DE.S]
        result[j+2] = u[b+DE.E0] + u[b+DE.E1]
        result[j+3] = u[b+DE.L0] + u[b+DE.L1]

        result[j+4] = sum(
            u[b+c] for c in DE.I0SP:DE.I1EP
        )

        result[j+5] = u[b+DE.T]
        result[j+6] = u[b+DE.R]
    end

    return result
end


const COUNTER_PAIRS = (
    TB.CumPrimaryInfections => DE.CumPrimaryInfections,
    TB.CumReinfections => DE.CumReinfections,
    TB.CumFastProgression => DE.CumFastProgression,
    TB.CumSlowProgression => DE.CumSlowProgression,
    TB.CumRelapse => DE.CumRelapse,
    TB.CumTreatmentInitiation => DE.CumTreatmentInitiation,
    TB.CumTreatmentCompletion => DE.CumTreatmentCompletion,
    TB.CumUntreatedTBDeaths => DE.CumUntreatedTBDeaths,
    TB.CumTreatmentTBDeaths => DE.CumTreatmentTBDeaths,
    TB.CumSelfCure => DE.CumSelfCure,
)

function check_counters(
    old_u, new_u, old_model, new_model;
    atol=1e-5, rtol=1e-5,
)
    os = nstates(state_layout(old_model))
    ns = nstates(state_layout(new_model))

    for a in 1:state_layout(old_model).nages
        ob = (a - 1) * os
        nb = (a - 1) * ns

        for (old_c, new_c) in COUNTER_PAIRS
            @test isapprox(
                old_u[ob+old_c],
                new_u[nb+new_c];
                atol=atol,
                rtol=rtol,
            )
        end
    end
end


# ------------------------------------------------------------
# Test 1: original Dodd equivalence
# ------------------------------------------------------------

@testset "Original Dodd equivalence" begin

    nages = 30
    contact = Matrix{Float64}(I, nages, nages)

    # All active disease is smear-positive pulmonary.
    presentation = zeros(nages, 3)
    presentation[:, 1] .= 1.0

    original = DoddTBModel(
        contact;
        pulmonary_fraction=1.0,
    )

    extended = DoddExtendedTBModel(
        contact,
        presentation;
        reinfection=true,
    )

    @test original.contact ≈ extended.reference.contact

    theta = merge(
        map(median, priors(extended)),
        extended_fixed_parameters(),
        (psi=0.15,),
    )

    p_original = make_parameters(original, theta)
    p_extended = make_parameters(extended, theta)

    u_original = synthetic_initial_state(original)
    u_extended = extend_initial_state(
        extended, u_original
    )

    # Same population distribution after aggregation.
    @test collapse_extended(u_extended, extended) ≈
        original_epi(u_original, original)

    # Compare derivatives at identical initial conditions.
    du_original = rhs(
        original, p_original, u_original
    )

    du_extended = rhs(
        extended, p_extended, u_extended
    )

    @test isapprox(
        original_epi(du_original, original),
        collapse_extended(du_extended, extended);
        atol=1e-9,
        rtol=1e-9,
    )

    check_counters(
        du_original, du_extended,
        original, extended;
        atol=1e-9, rtol=1e-9,
    )

    # Compare integrated trajectories.
    sol_original = solve_without_demography(
        original, p_original, u_original
    )

    sol_extended = solve_without_demography(
        extended, p_extended, u_extended
    )

    @test successful_retcode(sol_original)
    @test successful_retcode(sol_extended)

    @test length(sol_original.u) == length(sol_extended.u)

    for j in eachindex(sol_original.u)
        @test isapprox(
            original_epi(sol_original.u[j], original),
            collapse_extended(sol_extended.u[j], extended);
            atol=1e-5,
            rtol=1e-5,
        )

        check_counters(
            sol_original.u[j],
            sol_extended.u[j],
            original,
            extended,
        )
    end

    println("Original Dodd equivalence passed.")
end


# ------------------------------------------------------------
# Test 2: reinfection = false versus psi = 0
# ------------------------------------------------------------

@testset "Zero-reinfection equivalence" begin

    nages = 30
    contact = Matrix{Float64}(I, nages, nages)

    presentation = repeat(
        reshape([0.5, 0.3, 0.2], 1, 3),
        nages, 1,
    )

    full = DoddExtendedTBModel(
        contact, presentation;
        reinfection=true,
    )

    reduced = DoddExtendedTBModel(
        contact, presentation;
        reinfection=false,
    )

    theta = merge(
        map(median, priors(full)),
        extended_fixed_parameters(),
        (psi=0.0,),
    )

    p_full = make_parameters(full, theta)
    p_reduced = make_parameters(reduced, theta)

    @test p_full.base.psi == 0.0
    @test p_reduced.base.psi == 0.0
    @test !haskey(priors(reduced), :psi)

    original = synthetic_initial_state(full.reference)
    u0 = extend_initial_state(full, original)

    # Give some individuals a previous-treatment history.
    stride = nstates(state_layout(full))

    for a in 1:nages
        b = (a - 1) * stride

        l = 0.25 * u0[b+DE.L0]
        u0[b+DE.L0] -= l
        u0[b+DE.L1] += l

        for k in 0:2
            i = 0.25 * u0[b+DE.I0SP+k]
            u0[b+DE.I0SP+k] -= i
            u0[b+DE.I1SP+k] += i
        end
    end

    @test isapprox(
        rhs(full, p_full, u0),
        rhs(reduced, p_reduced, u0);
        atol=1e-10, rtol=1e-10,
    )

    sol_full = solve_without_demography(
        full, p_full, u0
    )

    sol_reduced = solve_without_demography(
        reduced, p_reduced, u0
    )

    @test successful_retcode(sol_full)
    @test successful_retcode(sol_reduced)

    for j in eachindex(sol_full.u)
        @test isapprox(
            sol_full.u[j],
            sol_reduced.u[j];
            atol=1e-6,
            rtol=1e-6,
        )
    end

    # Without reinfection, an initially empty E1 stays empty.
    for u in sol_full.u
        @test maximum(
            abs(u[(a-1)*stride+DE.E1])
            for a in 1:nages
        ) < 1e-8
    end

    println("Zero-reinfection equivalence passed.")
end
