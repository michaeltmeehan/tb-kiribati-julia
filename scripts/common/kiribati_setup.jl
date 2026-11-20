
using TBKiribatiJulia
using CSV
using DataFrames

include(joinpath(@__DIR__, "kiribati_paths.jl"))

"""
    setup_dodd_analysis(;
        start_year=1970,
        end_year=2025,
        output_years=2013:2024,
        nages=96,
    )

Load Kiribati inputs and construct the Dodd TB model
and projected demography.

Returns a NamedTuple containing:
- model and demography
- raw and structured notification data
- age-group definitions
- simulation and observation years
- input and output directories
"""
function setup_dodd_analysis(;
    start_year::Int=1970,
    end_year::Int=2025,
    output_years=2013:2024,
    nages::Int=96,
    use_historical_notifications::Bool=false,
    national_start_year::Int=1980,
)
    # --------------------------------------------------------
    # 1. Paths
    # --------------------------------------------------------

    root = normpath(joinpath(@__DIR__, "..", ".."))

    paths = kiribati_data_paths(root)

    data_dir = paths.data
    surveillance_dir = paths.surveillance
    output_dir = joinpath(root, "results", "Dodd")

    demographic_dir = paths.legacy_demography
    contact_file = paths.legacy_contact

    notification_file = joinpath(
        surveillance_dir,
        "tb_notifications_by_age.csv",
    )

    # --------------------------------------------------------
    # 2. Configuration
    # --------------------------------------------------------

    age_breaks = [
        0, 5, 15, 25, 35, 45, 55, 65, 96
    ]

    age_labels = [
        "0-4", "5-14", "15-24", "25-34",
        "35-44", "45-54", "55-64", "65+",
    ]

    years = collect(output_years)

    start_year < end_year ||
        throw(ArgumentError(
            "start_year must precede end_year"
        ))

    all(y -> start_year <= y < end_year, years) ||
        throw(ArgumentError(
            "Output years must lie within simulation period"
        ))

    last(age_breaks) == nages ||
        throw(ArgumentError(
            "Age-group definitions require nages=96"
        ))

    # --------------------------------------------------------
    # 3. Load demographic and contact data
    # --------------------------------------------------------

    demographic_data = load_demographic_data(
        demographic_dir
    )

    demographic_data.first_year <= start_year ||
        throw(ArgumentError(
            "Demographic data start too late"
        ))

    demographic_data.last_year >= end_year ||
        throw(ArgumentError(
            "Demographic data end too early"
        ))

    contact = load_contact_matrix(
        contact_file;
        nages=nages,
    )

    # --------------------------------------------------------
    # 4. Load surveillance data
    # --------------------------------------------------------

    notifications = CSV.read(
        notification_file,
        DataFrame,
    )

    surveillance = load_kiribati_surveillance(
        surveillance_dir;
        age_labels=age_labels,
        age_years=years,
    )

    age_observations = surveillance.age_notifications

    observations = if use_historical_notifications
        combine_notification_observations(
            age_observations,
            surveillance.national_notifications;
            n_age_groups=length(age_labels),
            start_year=start_year,
            end_year=end_year,
            national_start_year=national_start_year,
        )
    else
        age_observations
    end

    # --------------------------------------------------------
    # 5. Construct model and demography
    # --------------------------------------------------------

    model = DoddTBModel(contact)

    demography = ProjectedDemography(
        demographic_data;
        nages=nages,
    )

    # --------------------------------------------------------
    # 6. Return analysis inputs
    # --------------------------------------------------------

    return (
        model=model,
        demography=demography,
        demographic_data=demographic_data,
        contact=contact,
        notifications=notifications,
        surveillance=surveillance,
        age_observations=age_observations,
        calibration_years=[obs.year for obs in observations],
        observations=observations,
        age_breaks=age_breaks,
        age_labels=age_labels,
        start_year=start_year,
        end_year=end_year,
        output_years=years,
        data_dir=data_dir,
        output_dir=output_dir,
        paths=paths,
        surveillance_dir=surveillance_dir,
        demographic_dir=demographic_dir,
        contact_file=contact_file,
    )
end
