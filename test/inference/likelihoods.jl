
using Test
using TBKiribatiJulia
using Distributions: NegativeBinomial, Binomial, logpdf
using Statistics: mean, var

const TB = TBKiribatiJulia

@testset "Inference likelihood definitions" begin

    @testset "NB2 parameterisation" begin
        mu = 12.0
        sigma = 0.35
        y = 9

        phi = inv(sigma^2)
        p = phi / (phi + mu)

        reference = NegativeBinomial(phi, p)

        @test TB.nb2_loglikelihood(y, mu, sigma) ≈
            logpdf(reference, y)

        @test mean(reference) ≈ mu
        @test var(reference) ≈
            mu + sigma^2 * mu^2

        @test TB.nb2_loglikelihood(0, 0.0, sigma) == 0
        @test TB.nb2_loglikelihood(1, 0.0, sigma) == -Inf
        @test TB.nb2_loglikelihood(-1, mu, sigma) == -Inf
        @test TB.nb2_loglikelihood(y, -1.0, sigma) == -Inf
        @test TB.nb2_loglikelihood(y, mu, 0.0) == -Inf
    end

    @testset "Independent observations" begin
        obs = TB.NB2Observation(nothing)

        rows = [
            (observed=8, expected=9.5),
            (observed=12, expected=11.2),
        ]

        theta = (sigma_obs=0.3,)

        expected = sum(
            TB.nb2_loglikelihood(
                r.observed, r.expected, theta.sigma_obs
            )
            for r in rows
        )

        @test TB.observation_year_loglikelihood(
            obs, rows, theta
        ) ≈ expected

        @test TB.observation_variance(
            obs, 10.0, theta
        ) ≈ 19.0
    end

    @testset "Shared annual random effects" begin
        sigma = 0.3
        tau = 0.25

        obs = TB.AnnualRandomEffectNB2Observation(
            nothing, nothing;
            quadrature_order=30,
        )

        rows = [
            (observed=8, expected=9.5),
            (observed=12, expected=11.2),
        ]

        theta = (
            sigma_obs=sigma,
            tau_year=tau,
        )

        @test sum(obs.weights) ≈ 1.0 atol=1e-12
        @test all(obs.weights .>= 0)

        # Independently evaluate the weighted mixture.
        mixture = sum(
            obs.weights[k] * prod(
                exp(TB.nb2_loglikelihood(
                    r.observed,
                    r.expected * exp(
                        tau * obs.nodes[k] - tau^2 / 2
                    ),
                    sigma,
                ))
                for r in rows
            )
            for k in eachindex(obs.nodes)
        )

        actual = TB.observation_year_loglikelihood(
            obs, rows, theta
        )

        @test actual ≈ log(mixture) atol=1e-10

        # Exact reduction when the annual-effect SD is zero.
        theta_zero = (
            sigma_obs=sigma,
            tau_year=0.0,
        )

        independent = TB.observation_year_loglikelihood(
            TB.NB2Observation(nothing),
            rows,
            theta_zero,
        )

        @test TB.observation_year_loglikelihood(
            obs, rows, theta_zero
        ) ≈ independent

        mu = 10.0

        @test TB.observation_variance(
            obs, mu, theta
        ) ≈ mu + mu^2 * (
            (1 + sigma^2) * exp(tau^2) - 1
        )

        @test TB.observation_year_loglikelihood(
            obs,
            rows,
            (sigma_obs=sigma, tau_year=-0.1),
        ) == -Inf
    end

    @testset "Treatment-death likelihood" begin
        rows = [
            (year=2023, cohort_size=100, died=12),
            (year=2024, cohort_size=90, died=10),
            (year=2025, cohort_size=110, died=15),
        ]

        obs = TB.TreatmentDeathObservation(
            rows;
            treatment_duration=0.5,
            background_hazard=0.1,
        )

        # The default excludes the 2024 cohort.
        @test length(obs.rows) == 2
        @test [r.year for r in obs.rows] == [2023, 2025]

        theta = (treatment_cfr=0.2,)

        rate = 1 / 0.5
        expected_p = (
            0.2 * rate + 0.1
        ) / (rate + 0.1)

        @test TB.treatment_death_probability(
            obs, theta
        ) ≈ expected_p

        expected_ll = sum(
            logpdf(
                Binomial(r.cohort_size, expected_p),
                r.died,
            )
            for r in obs.rows
        )

        @test TB.treatment_death_loglikelihood(
            obs, theta
        ) ≈ expected_ll

        # Without background mortality, the competing
        # risks death probability equals treatment_cfr.
        obs_zero = TB.TreatmentDeathObservation(
            rows;
            background_hazard=0.0,
        )

        @test TB.treatment_death_probability(
            obs_zero, theta
        ) ≈ 0.2

        @test TB.treatment_death_loglikelihood(
            obs, (treatment_cfr=1.1,)
        ) == -Inf

        @test TB.treatment_death_loglikelihood(
            nothing, theta
        ) == 0.0

        @test_throws ArgumentError TB.TreatmentDeathObservation(
            [(year=2023, cohort_size=5, died=6)]
        )
    end

    @testset "Conditional recurrence likelihood" begin
        national = [
            (year=2023, cases=90),
            (year=2024, cases=70),
        ]

        recurrent = [
            (year=2023, cases=18),
            (year=2024, cases=14),
        ]

        obs = TB.RecurrentTBObservation(
            recurrent, national
        )

        output = (
            years=[2023, 2024],
            notifications=[
                40.0 30.0
                60.0 50.0
            ],
            recurrent_notifications=[
                8.0 6.0
                12.0 10.0
            ],
        )

        # Both years have predicted recurrence fraction 0.2.
        expected = logpdf(Binomial(90, 0.2), 18) +
            logpdf(Binomial(70, 0.2), 14)

        @test TB.recurrence_loglikelihood(
            obs, output
        ) ≈ expected

        @test TB.recurrence_loglikelihood(
            nothing, output
        ) == 0.0

        @test_throws ArgumentError TB.recurrence_loglikelihood(
            obs,
            (years=output.years,
                notifications=output.notifications),
        )

        invalid = merge(
            output,
            (
                recurrent_notifications=
                output.notifications .* 1.1,
            ),
        )

        @test TB.recurrence_loglikelihood(
            obs, invalid
        ) == -Inf

        @test_throws ArgumentError TB.RecurrentTBObservation(
            [(year=2023, cases=100)],
            national,
        )
    end


    @testset "Quadrature with impossible nodes" begin
        rows = [
            (observed=8, expected=10.0),
        ]

        theta = (
            sigma_obs=0.3,
            tau_year=1.0,
        )

        # First node: finite likelihood.
        # Second node: exp(799.5) overflows, giving
        # zero likelihood at that node.
        obs = TB.AnnualRandomEffectNB2Observation(
            nothing,
            nothing,
            [0.0, 800.0],
            [0.5, 0.5],
        )

        expected = log(0.5) +
            TB.nb2_loglikelihood(
            8,
            10.0 * exp(-0.5),
            0.3,
        )

        actual = TB.observation_year_loglikelihood(
            obs, rows, theta
        )

        @test isfinite(actual)
        @test actual ≈ expected atol=1e-12

        # No node can explain a positive observed
        # count when the expected count is zero.
        impossible_rows = [
            (observed=1, expected=0.0),
        ]

        @test TB.observation_year_loglikelihood(
            obs, impossible_rows, theta
        ) == -Inf
    end


    @testset "Annual-effect likelihood differentiation" begin
        using ForwardDiff

        obs = TB.AnnualRandomEffectNB2Observation(
            nothing,
            nothing;
            quadrature_order=30,
        )

        rows = [
            (observed=8, expected=9.5),
            (observed=12, expected=11.2),
        ]

        function loglik(tau)
            return TB.observation_year_loglikelihood(
                obs,
                rows,
                (
                    sigma_obs=0.3,
                    tau_year=tau,
                ),
            )
        end

        tau0 = 0.25
        h = 1e-5

        grad_ad = ForwardDiff.derivative(
            loglik, tau0
        )

        grad_fd = (
            loglik(tau0 + h) -
                loglik(tau0 - h)
        ) / (2h)

        @test isfinite(grad_ad)
        @test isapprox(
            grad_ad, grad_fd;
            rtol=1e-4,
            atol=1e-5,
        )
    end

    @testset "Conditional pulmonary / extrapulmonary likelihood" begin
        TB = TBKiribatiJulia

        obs = TB.SiteTBObservation([
            (year=2023, pulmonary=70, extrapulmonary=30, total=100),
            (year=2024, pulmonary=24, extrapulmonary=16, total=40),
        ])

        output = (
            years=[2023, 2024],
            pulmonary_notifications=[20.0 6.0; 50.0 18.0],
            extrapulmonary_notifications=[10.0 4.0; 20.0 12.0],
        )

        expected = logpdf(Binomial(100, 0.7), 70) +
            logpdf(Binomial(40, 0.6), 24)

        @test TB.site_loglikelihood(obs, output) ≈ expected
        @test TB.site_loglikelihood(nothing, output) == 0.0

        @test_throws ArgumentError TB.SiteTBObservation([
            (year=2023, pulmonary=70, extrapulmonary=31, total=100),
        ])

        @test_throws ArgumentError TB.SiteTBObservation([
            (year=2023, pulmonary=70, extrapulmonary=30, total=100),
            (year=2023, pulmonary=70, extrapulmonary=30, total=100),
        ])

        @test_throws ArgumentError TB.site_loglikelihood(obs, (
            years=[2023, 2024],
            pulmonary_notifications=output.pulmonary_notifications,
        ))

        @test_throws ArgumentError TB.site_loglikelihood(
            obs,
            merge(output, (years=[2025, 2024],)),
        )

        invalid = merge(output, (
            extrapulmonary_notifications=fill(-1.0, 2, 2),
        ))

        @test TB.site_loglikelihood(obs, invalid) == -Inf
    end

end
