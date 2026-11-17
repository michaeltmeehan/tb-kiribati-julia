
using Distributions: LogNormal

struct DoddExtendedTBParameters
    base::DoddTBParameters

    # Age-specific fractions at active disease onset.
    # Columns: smear-positive, smear-negative, extrapulmonary.
    presentation::Matrix{Float64}

    # Relative infectiousness of the three presentations.
    relative_infectiousness::NTuple{3,Float64}

    # Relative detection hazards.
    relative_detection::NTuple{3,Float64}
end


"""
Additional priors for the extended model.

These are provisional modelling priors rather than
estimates obtained from Kiribati data.

Initially, fix the corresponding parameters during
calibration using extended_fixed_parameters().
"""
function priors(model::DoddExtendedTBModel)
    base = priors(model.reference)

    if !model.reinfection
        names = Tuple(
            name for name in keys(base)
                     if name != :psi
        )

        base = NamedTuple{names}(
            Tuple(getproperty(base, name) for name in names)
        )
    end

    additional = (
        smear_negative_infectiousness=
        LogNormal(log(0.35), 0.4),
        smear_negative_detection=
        LogNormal(0.0, 0.5),
        extrapulmonary_detection=
        LogNormal(0.0, 0.5),
    )

    return merge(base, additional)
end


"""
Initial fixed values for the additional parameters.
"""
extended_fixed_parameters() = (
    smear_negative_infectiousness=0.35,
    smear_negative_detection=1.0,
    extrapulmonary_detection=1.0,
)


function make_parameters(
    model::DoddExtendedTBModel,
    draw::NamedTuple,
)
    # Reinfection can be completely excluded from the
    # reduced model without retaining an unused psi prior.
    base_draw = model.reinfection ?
                draw :
                merge(draw, (psi=0.0,))

    base = make_parameters(
        model.reference,
        base_draw,
    )

    q_negative =
        Float64(draw.smear_negative_infectiousness)

    d_negative =
        Float64(draw.smear_negative_detection)

    d_extra =
        Float64(draw.extrapulmonary_detection)

    all(x -> isfinite(x) && x > 0, (
        q_negative,
        d_negative,
        d_extra,
    )) || throw(ArgumentError(
        "Invalid presentation-specific parameters"
    ))

    return DoddExtendedTBParameters(
        base,
        model.presentation,
        (1.0, q_negative, 0.0),
        (1.0, d_negative, d_extra),
    )
end
