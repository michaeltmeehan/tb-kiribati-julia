
let
    root = normpath(joinpath(@__DIR__, "..", ".."))

    source = joinpath(
        root, "src", "models", "Ragonnet",
        "legacy_dynamics.jl"
    )

    destination = joinpath(
        root, "src", "models", "Ragonnet",
        "dynamics.jl"
    )

    text = read(source, String)

    # Retain only the original mathematical kernel.
    marker = "function compute_force_of_infection!("
    pos = findfirst(marker, text)
    isnothing(pos) && error("Legacy FOI function not found")

    text = text[first(pos):end]

    # Replace the function interfaces.
    old_foi = "function compute_force_of_infection!(λ::AbstractVector{<:Real}, u::AbstractVector, p::TBParams)"
    new_foi = "function ragonnet_force_of_infection!(λ::AbstractVector{<:Real}, u::AbstractVector, p::RagonnetKernel.TBParams, layout::StateLayout)"

    old_rhs = "function tb_rhs!(du, u, p::TBParams, t)"
    new_rhs = "function ragonnet_rhs!(du, u, p::RagonnetKernel.TBParams, t, layout::StateLayout)"

    occursin(old_foi, text) || error("Unexpected FOI signature")
    occursin(old_rhs, text) || error("Unexpected RHS signature")

    text = replace(text, old_foi => new_foi)
    text = replace(text, old_rhs => new_rhs)

    # Update the internal function call.
    text = replace(
        text,
        "compute_force_of_infection!(λ, u, p)" => "ragonnet_force_of_infection!(λ, u, p, layout)"
    )

    # Replace global dimensions with StateLayout.
    for (old, new) in (
        "NAGE" => "layout.nages",
        "NEPI" => "layout.nepi",
        "NSTATE" => "nstates(layout)",
    )
        text = replace(
            text,
            Regex("\\b" * old * "\\b") => new
        )
    end

    # Qualify the 17 state indices.
    indices = (
        "MtbNaive", "Contained", "Cleared",
        "Recovered", "Incipient",
        "SubClinLow", "SubClinInf",
        "ClinLow", "ClinInf", "Treatment",
        "CumInfectionsOther",
        "CumInfectionsContained",
        "CumProgressionToActiveTB",
        "CumTreatmentInitiation",
        "CumTreatmentCompletion",
        "CumRelapseTB",
        "CumDeathsTB",
    )

    for name in indices
        text = replace(
            text,
            Regex("\\b" * name * "\\b") => "RagonnetIndices.$name"
        )
    end

    # Check that no legacy dimension constants remain.
    for name in ("NAGE", "NEPI", "NSTATE")
        occursin(Regex("\\b" * name * "\\b"), text) &&
            error("Unconverted dimension: $name")
    end

    isfile(destination) &&
        error("Refusing to overwrite existing $destination")

    header = """
    # Ragonnet differential equations.
    # Migrated from legacy_dynamics.jl.
    # State indices are model-specific.
    # Dimensions are obtained from StateLayout.
    # Equations and parameter values are unchanged.

    """

    write(destination, header * text)

    println("Generated: ", destination)
end
