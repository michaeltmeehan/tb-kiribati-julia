
using Test
using TBKiribatiJulia

@testset "TBKiribatiJulia model domain" begin

    @testset "Interfaces and numerical infrastructure" begin
        include("models/interfaces.jl")
        include("models/numerical_infrastructure.jl")
        include("models/mixing.jl")
    end

    @testset "Demography" begin
        include("models/demography.jl")
        include("models/cohort_demography.jl")
        include("models/cohort_wpp_integration.jl")
    end

    @testset "Model implementations" begin
        include("models/dodd_extended.jl")
        include("models/ragonnet.jl")
    end

    @testset "Simulation" begin
        include("models/simulation.jl")
        include("models/prepared_simulation.jl")
        include("models/equilibrium.jl")
        include("models/end_to_end.jl")
        include("models/ensemble.jl")
    end

    @testset "Outputs" begin
        include("models/outputs.jl")
        include("models/output_definitions.jl")
    end

    @testset "Inference" begin
        include("inference/likelihoods.jl")
        include("inference/calibration_foundation.jl")
        include("inference/optimisation.jl")
    end
end
