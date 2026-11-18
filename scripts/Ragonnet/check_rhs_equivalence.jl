
using TBKiribatiJulia
using Test
using Random
using LinearAlgebra

# Load the original source in a separate namespace.
module RagonnetReference

using DelimitedFiles
using LinearAlgebra: mul!

const ROOT = normpath(
    joinpath(@__DIR__, "..", "..")
)

data_path(name) = joinpath(ROOT, "data", name)

include(joinpath(ROOT, "src", "parameters.jl"))
include(joinpath(ROOT, "src", "model.jl"))

end

const TB = TBKiribatiJulia
const REF = RagonnetReference

rng = MersenneTwister(20261118)

@testset "Ragonnet RHS equivalence" begin

    # Use the exact original contact matrix.
    C = copy(REF.CONTACT)

    model = TB.RagonnetTBModel(C)

    @test TB.state_layout(model).nepi == 10
    @test TB.state_layout(model).ncum == 7
    @test TB.state_layout(model).nages == 96
    @test model.contact == C

    idx = TB.RagonnetIndices
    layout = TB.state_layout(model)

    @test TB.nstates(layout) == 17
    @test layout.nepi == 10
    @test layout.ncum == 7
    @test layout.nages == 96

    @test (
        idx.MtbNaive,
        idx.Contained,
        idx.Cleared,
        idx.Recovered,
        idx.Incipient,
        idx.SubClinLow,
        idx.SubClinInf,
        idx.ClinLow,
        idx.ClinInf,
        idx.Treatment,
    ) == Tuple(1:10)

    @test (
        idx.CumInfectionsOther,
        idx.CumInfectionsContained,
        idx.CumProgressionToActiveTB,
        idx.CumTreatmentInitiation,
        idx.CumTreatmentCompletion,
        idx.CumRelapseTB,
        idx.CumDeathsTB,
    ) == Tuple(11:17)

    for age in 1:96
        for compartment in 1:17
            @test TB.state_index(
                layout, age, compartment
            ) == (age - 1)*17 + compartment
        end
    end


    @testset "Treatment parameter construction" begin
        scenarios = (
            (;),
            (; tx_period=0.25,
                tx_success_prop=0.65,
                pct_neg_tx_death=0.3),
            (; tx_recovery_rate=1.7),
            (; tx_relapse_rate=0.2),
            (; tx_death_rate=0.05),
            (; tx_recovery_rate=1.1,
                tx_relapse_rate=0.15,
                tx_death_rate=0.07),
        )

        for kwargs in scenarios
            old = REF.make_parameters(C; kwargs...)
            new = TB.make_parameters(model; kwargs...)

            for field in (
                :tx_recovery_rate,
                :tx_relapse_rate,
                :tx_death_rate,
            )
                @test getfield(new, field) ==
                    getfield(old, field)
            end
        end
    end


    for i in 1:100

        # Vary model parameters across comparisons.
        kwargs = (
            beta=0.0001 + 0.003 * rand(rng),
            detection_rate=0.1 + 2 * rand(rng),
            clearance_rate=0.01 + 0.2 * rand(rng),
            progression_child=0.5 + 3 * rand(rng),
            progression_15_64=0.05 + rand(rng),
            prop_infectious=rand(rng),
            disease_mortality_clin_inf=0.4 * rand(rng),
        )

        p_old = REF.make_parameters(C; kwargs...)
        p_new = TB.make_parameters(model; kwargs...)

        # Every scientific and derived parameter must match.
        legacy_fields = setdiff(
            fieldnames(typeof(p_old)),
            (:tmp_q, :tmp_foi, :tmp_age_pop),
        )

        @test Set(fieldnames(typeof(p_new))) ==
            Set(legacy_fields)

        for field in legacy_fields
            @test isequal(
                getfield(p_new, field),
                getfield(p_old, field),
            )
        end

        @test p_new isa TB.RagonnetTBParameters
        @test !hasproperty(p_new, :tmp_q)
        @test !hasproperty(p_new, :tmp_foi)
        @test !hasproperty(p_new, :tmp_age_pop)

        # Valid nonnegative living states, with counters
        # independently randomised.
        u = rand(rng, 96 * 17) .* 100

        du_old = zeros(length(u))
        du_new = zeros(length(u))

        REF.tb_rhs!(du_old, u, p_old, 2023.0)

        ctx = TB.SimulationContext(
            model,
            p_new,
            nothing,
            TB.TBWorkspace(96),
        )

        TB.tb_rhs!(du_new, u, ctx, 2023.0)

        @test isapprox(
            du_new, du_old;
            rtol=1e-13,
            atol=1e-12,
        )

        # Living-population derivatives must equal
        # negative TB mortality, age by age.
        D = reshape(du_new, 17, 96)

        for age in 1:96
            living_change = sum(@view D[1:10, age])
            tb_deaths = D[17, age]

            @test isapprox(
                living_change + tb_deaths,
                0.0;
                atol=1e-9,
            )
        end
    end
end
