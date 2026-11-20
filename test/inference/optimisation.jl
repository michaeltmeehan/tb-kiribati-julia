
using Test
using Optim
using TBKiribatiJulia

# Run after calibration_foundation.jl,
# which defines CalibrationFixture.

@testset "Calibration optimisers" begin
    TB = TBKiribatiJulia
    F = CalibrationFixture

    problem = F.problem()

    initial = (
        beta=0.15,
        sigma_obs=0.3,
    )

    fixed = (sigma_obs=0.3,)

    @testset "L-BFGS versus Nelder–Mead" begin
        nm = TB.fit_mle(
            problem;
            optimizer=:nelder_mead,
            initial,
            fixed,
            nstarts=1,
            iterations=200,
        )

        lbfgs = TB.fit_mle(
            problem;
            optimizer=:lbfgs,
            initial,
            fixed,
            nstarts=1,
            iterations=200,
        )

        @test isfinite(nm.log_likelihood)
        @test isfinite(lbfgs.log_likelihood)

        @test lbfgs.converged

        @test isapprox(
            lbfgs.log_likelihood,
            nm.log_likelihood;
            atol=1e-4,
        )

        @test isapprox(
            lbfgs.parameters.beta,
            nm.parameters.beta;
            rtol=1e-3,
        )

        @test lbfgs.parameters.sigma_obs == 0.3
        @test lbfgs.n_parameters == 1
    end

    @testset "MAP optimisation" begin
        result = TB.fit_map(
            problem;
            optimizer=:lbfgs,
            initial,
            fixed,
            nstarts=1,
            iterations=200,
        )

        @test isfinite(result.log_posterior)

        @test result.log_posterior ≈
            result.log_prior +
            result.log_likelihood

        @test result.parameters.sigma_obs == 0.3
    end

    @testset "Invalid optimiser" begin
        @test_throws ArgumentError begin
            TB.fit_mle(
                problem;
                optimizer=:unknown,
                nstarts=1,
            )
        end
    end
end
