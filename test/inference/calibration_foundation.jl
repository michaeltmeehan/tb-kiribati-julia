
using Test
using ForwardDiff
using Distributions: LogNormal
using TBKiribatiJulia

module CalibrationFixture

import ..TBKiribatiJulia
using Distributions: LogNormal

const TB = TBKiribatiJulia

struct TestModel <: TB.AbstractTBModel
    layout::TB.StateLayout
end

struct TestDemography <: TB.AbstractDemography end

TB.state_layout(m::TestModel) = m.layout

TB.priors(::TestModel) = (
    beta=LogNormal(log(0.2), 0.5),
)

TB.make_parameters(::TestModel, draw::NamedTuple) = draw

function TB.initial_state(
    ::TestModel,
    params,
    ::TestDemography;
    year,
)
    return [
        100 * one(params.beta),
        zero(params.beta),
    ]
end

function TB.tb_rhs!(
    du,
    u,
    ctx::TB.SimulationContext{<:TestModel},
    t,
)
    flow = ctx.params.beta * u[1]
    du[1] = -flow
    du[2] = flow
    return nothing
end

function TB.apply_demography!(
    u::AbstractVector,
    ::TestModel,
    ::TestDemography,
    year::Real;
    kwargs...,
)
    return nothing
end

function TB.model_output(
    ::TestModel,
    ex::TB.OutputExtraction,
    ::Val{:notifications},
)
    return TB.counter_output(ex, 2)
end

model() = TestModel(TB.StateLayout(1, 1, 1))
demography() = TestDemography()

observations() = [
    (
        year=2023,
        labels=["all"],
        counts=[18],
        groups=[[1]],
    ),
    (
        year=2024,
        labels=["all"],
        counts=[15],
        groups=[[1]],
    ),
]

function problem()
    return TB.CalibrationProblem(
        model(),
        demography(),
        observations();
        start_year=2023,
        end_year=2025,
        age_breaks=[0, 1],
        observation_model=TB.NB2Observation(
            LogNormal(log(0.3), 0.2),
        ),
        abstol=1e-11,
        reltol=1e-10,
    )
end

theta(beta; sigma_obs=0.3) = (; beta, sigma_obs)

expected(beta) = [
    100 * (1 - exp(-beta)),
    100 * exp(-beta) * (1 - exp(-beta)),
]

end # module


@testset "Calibration foundation" begin
    TB = TBKiribatiJulia
    F = CalibrationFixture

    problem = F.problem()

    @testset "Prepared configuration" begin
        @test problem.simulation isa TB.PreparedSimulation
        @test length(typeof(problem).parameters) == 6
        @test problem.age_breaks == [0, 1]
        @test problem.start_year == 2023
        @test problem.end_year == 2025
    end

    @testset "Analytical prediction" begin
        for beta in (0.1, 0.2, 0.4)
            out = TB.predict(problem, F.theta(beta))

            @test out.years == [2023, 2024]
            @test size(out.notifications) == (1, 2)

            @test vec(out.notifications) ≈
                F.expected(beta) atol=1e-7
        end
    end

    @testset "Likelihood identity" begin
        theta = F.theta(0.2)
        out = TB.predict(problem, theta)

        expected_ll = sum(
            TB.nb2_loglikelihood(y, mu, theta.sigma_obs)
            for (y, mu) in zip(
                (18, 15),
                vec(out.notifications),
            )
        )

        @test TB.calibration_loglikelihood(
            problem, theta; output=out
        ) ≈ expected_ll

        @test TB.calibration_loglikelihood(
            problem, theta
        ) ≈ expected_ll

        @test isfinite(
            TB.calibration_logposterior(problem, theta)
        )
    end

    @testset "Repeated evaluations" begin
        first = TB.predict(problem, F.theta(0.1))

        TB.predict(problem, F.theta(0.4))

        repeated = TB.predict(problem, F.theta(0.1))

        @test first.notifications ≈ repeated.notifications
    end

    @testset "Prediction gradient" begin
        f(beta) = sum(
            TB.predict(problem, F.theta(beta)).notifications
        )

        beta0 = 0.2
        h = 1e-5

        ad = ForwardDiff.derivative(f, beta0)
        fd = (f(beta0+h) - f(beta0-h)) / (2h)

        @test isfinite(ad)
        @test isapprox(ad, fd; rtol=1e-4, atol=1e-5)
    end

    @testset "Complete likelihood gradient" begin
        f(beta) = TB.calibration_loglikelihood(
            problem, F.theta(beta)
        )

        beta0 = 0.2
        h = 1e-5

        ad = ForwardDiff.derivative(f, beta0)
        fd = (f(beta0+h) - f(beta0-h)) / (2h)

        @test isfinite(ad)
        @test isapprox(ad, fd; rtol=1e-4, atol=1e-5)
    end

    @testset "Observation parameter gradient" begin
        f(sigma) = TB.calibration_loglikelihood(
            problem,
            F.theta(0.2; sigma_obs=sigma),
        )

        sigma0 = 0.3
        h = 1e-5

        ad = ForwardDiff.derivative(f, sigma0)
        fd = (f(sigma0+h) - f(sigma0-h)) / (2h)

        @test isfinite(ad)
        @test isapprox(ad, fd; rtol=1e-4, atol=1e-5)
    end
end


@testset "Annual random-effect calibration" begin
    TB = TBKiribatiJulia
    F = CalibrationFixture

    obsmodel = TB.AnnualRandomEffectNB2Observation(
        LogNormal(log(0.3), 0.2),
        LogNormal(log(0.2), 0.3);
        quadrature_order=30,
    )

    problem = TB.CalibrationProblem(
        F.model(),
        F.demography(),
        F.observations();
        start_year=2023,
        end_year=2025,
        age_breaks=[0, 1],
        observation_model=obsmodel,
        abstol=1e-11,
        reltol=1e-10,
    )

    theta(beta, sigma, tau) = (
        beta=beta,
        sigma_obs=sigma,
        tau_year=tau,
    )

    theta0 = theta(0.2, 0.3, 0.25)

    # Independent numerical reference for the marginal
    # likelihood, using the model's quadrature rule.
    function reference_loglikelihood(theta)
        mu = F.expected(theta.beta)
        observed = [18, 15]

        total = 0.0

        for j in eachindex(observed)
            mixture = sum(
                obsmodel.weights[k] *
                    exp(TB.nb2_loglikelihood(
                    observed[j],
                    mu[j] * exp(
                        theta.tau_year * obsmodel.nodes[k] -
                            theta.tau_year^2 / 2
                    ),
                    theta.sigma_obs,
                ))
                for k in eachindex(obsmodel.nodes)
            )

            total += log(mixture)
        end

        return total
    end

    @testset "Marginal likelihood" begin
        actual = TB.calibration_loglikelihood(
            problem, theta0
        )

        expected = reference_loglikelihood(theta0)

        @test isfinite(actual)
        @test isapprox(
            actual, expected;
            atol=1e-7,
            rtol=1e-8,
        )
    end

    @testset "Independent NB2 limit" begin
        theta_zero = theta(0.2, 0.3, 0.0)

        ll_random = TB.calibration_loglikelihood(
            problem, theta_zero
        )

        independent = TB.CalibrationProblem(
            F.model(),
            F.demography(),
            F.observations();
            start_year=2023,
            end_year=2025,
            age_breaks=[0, 1],
            observation_model=TB.NB2Observation(
                LogNormal(log(0.3), 0.2)
            ),
            abstol=1e-11,
            reltol=1e-10,
        )

        ll_independent =
            TB.calibration_loglikelihood(
                independent,
                (beta=0.2, sigma_obs=0.3),
            )

        @test isapprox(
            ll_random, ll_independent;
            atol=1e-9,
            rtol=1e-9,
        )
    end

    @testset "Complete likelihood gradients" begin
        h = 1e-5

        for parameter in (
            :beta,
            :sigma_obs,
            :tau_year,
        )
            x0 = getproperty(theta0, parameter)

            function objective(x)
                updated = merge(
                    theta0,
                    NamedTuple{(parameter,)}((x,)),
                )

                return TB.calibration_loglikelihood(
                    problem, updated
                )
            end

            gradient_ad =
                ForwardDiff.derivative(objective, x0)

            gradient_fd = (
                objective(x0 + h) -
                    objective(x0 - h)
            ) / (2h)

            @test isfinite(gradient_ad)
            @test isfinite(gradient_fd)

            @test isapprox(
                gradient_ad, gradient_fd;
                rtol=1e-3,
                atol=1e-4,
            )
        end
    end

    @testset "Repeated evaluations" begin
        ll1 = TB.calibration_loglikelihood(
            problem, theta0
        )

        TB.calibration_loglikelihood(
            problem, theta(0.4, 0.5, 0.1)
        )

        ll2 = TB.calibration_loglikelihood(
            problem, theta0
        )

        @test ll1 ≈ ll2
        @test isfinite(
            TB.calibration_logposterior(
                problem, theta0
            )
        )
    end
end
