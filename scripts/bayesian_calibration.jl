using TBKiribatiJulia

using Distributions
using Turing
using DynamicPPL
using StatsAPI
using FlexiChains

include("calibration_data.jl")


# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------

const PRIORS = (
    beta =
        LogNormal(log(1.0), 0.3),

    progression_child =
        LogNormal(log(1.5), 0.5),

    progression_5_14 =
        LogNormal(log(0.1), 0.5),

    progression_15_64 =
        LogNormal(log(0.2), 0.5),

    progression_65_plus =
        LogNormal(log(0.4), 0.5),
)


# Observation-model prior. sigma_obs controls extra-Poisson variation in the
# NB2 likelihood, with phi = 1 / sigma_obs^2.
const SIGMA_OBS_PRIOR =
    truncated(Normal(0.0, 0.3), 0.0, Inf)


# ---------------------------------------------------------------------------
# Bayesian calibration model
# ---------------------------------------------------------------------------

@model function tb_calibration_model()

    beta ~ PRIORS.beta

    progression_child ~
        PRIORS.progression_child

    progression_5_14 ~
        PRIORS.progression_5_14

    progression_15_64 ~
        PRIORS.progression_15_64

    progression_65_plus ~
        PRIORS.progression_65_plus

    sigma_obs ~
        SIGMA_OBS_PRIOR


    model_parameters = (
        beta =
            beta,

        progression_child =
            progression_child,

        progression_5_14 =
            progression_5_14,

        progression_15_64 =
            progression_15_64,

        progression_65_plus =
            progression_65_plus,
    )


    ll =
        calibration_loglikelihood(
            model_parameters;

            sigma_obs =
                sigma_obs,

            notification_observations =
                NOTIFICATION_OBSERVATIONS,

            relative_detection =
                RELATIVE_DETECTION,

            age_breaks =
                CALIBRATION_AGE_BREAKS,

            fixed_parameters =
                FIXED_PARAMETERS,

            tinit =
                TINIT,

            tfinal =
                TFINAL,
        )


    @addlogprob! ll

    return (
        model_parameters...,
        sigma_obs = sigma_obs,
    )
end


model =
    tb_calibration_model()


test_parameters = (
    beta = 1.0,
    progression_child = 1.5,
    progression_5_14 = 0.1,
    progression_15_64 = 0.2,
    progression_65_plus = 0.4,
)

test_sigma_obs = 0.3

ll =
    calibration_loglikelihood(
        test_parameters;
        sigma_obs = test_sigma_obs,
        notification_observations = NOTIFICATION_OBSERVATIONS,
        relative_detection = RELATIVE_DETECTION,
        age_breaks = CALIBRATION_AGE_BREAKS,
        fixed_parameters = FIXED_PARAMETERS,
        tinit = TINIT,
        tfinal = TFINAL,
    )

lp =
    sum(
        logpdf(prior, test_parameters[name])
        for (name, prior) in pairs(PRIORS)
    ) +
    logpdf(SIGMA_OBS_PRIOR, test_sigma_obs)

println()
println("Test parameter point:")
println("  log-likelihood = ", ll)
println("  log-prior      = ", lp)
println("  log-posterior  = ", ll + lp)

test_parameters_turing = (
    test_parameters...,
    sigma_obs = test_sigma_obs,
)

turing_ll =
    StatsAPI.loglikelihood(
        model,
        test_parameters_turing,
    )

turing_lp =
    DynamicPPL.logprior(
        model,
        test_parameters_turing,
    )

turing_joint =
    DynamicPPL.logjoint(
        model,
        test_parameters_turing,
    )


println()
println("Turing evaluation:")
println("  log-likelihood = ", turing_ll)
println("  log-prior      = ", turing_lp)
println("  log-joint      = ", turing_joint)

println()
println("Differences:")
println("  likelihood = ", turing_ll - ll)
println("  prior      = ", turing_lp - lp)
println("  joint      = ", turing_joint - (ll + lp))


# ---------------------------------------------------------------------------
# Pilot MCMC
# ---------------------------------------------------------------------------

using Random

const N_WALKERS = 10
const N_SAMPLES = 50

rng = Xoshiro(1234)

chain = sample(
    rng,
    model,
    Emcee(N_WALKERS),
    N_SAMPLES;
    progress = true,
)


println()
println("Pilot-chain summary:")
summarystats(chain)