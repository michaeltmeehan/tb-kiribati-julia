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
        LogNormal(log(1.0), 0.5),

    progression_child =
        LogNormal(log(1.5), 0.7),

    progression_5_14 =
        LogNormal(log(0.1), 0.8),

    progression_15_64 =
        LogNormal(log(0.2), 0.8),

    progression_65_plus =
        LogNormal(log(0.4), 0.7),
)


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

            notification_observations =
                NOTIFICATION_OBSERVATIONS,

            incidence_observations =
                INCIDENCE_OBSERVATIONS,

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

    return model_parameters
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

ll =
    calibration_loglikelihood(
        test_parameters;
        notification_observations = NOTIFICATION_OBSERVATIONS,
        incidence_observations = INCIDENCE_OBSERVATIONS,
        age_breaks = CALIBRATION_AGE_BREAKS,
        fixed_parameters = FIXED_PARAMETERS,
        tinit = TINIT,
        tfinal = TFINAL,
    )

lp =
    sum(
        logpdf(prior, test_parameters[name])
        for (name, prior) in pairs(PRIORS)
    )

println()
println("Test parameter point:")
println("  log-likelihood = ", ll)
println("  log-prior      = ", lp)
println("  log-posterior  = ", ll + lp)

turing_ll =
    StatsAPI.loglikelihood(
        model,
        test_parameters,
    )

turing_lp =
    DynamicPPL.logprior(
        model,
        test_parameters,
    )

turing_joint =
    DynamicPPL.logjoint(
        model,
        test_parameters,
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
const N_SAMPLES = 5_000

rng = Xoshiro(1234)

chain = sample(
    rng,
    model,
    Emcee(N_WALKERS),
    N_SAMPLES;
    progress = true,
)

println()
println(chain)

println()
println("Pilot-chain summary:")
summarystats(chain)