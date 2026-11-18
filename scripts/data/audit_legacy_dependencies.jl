
using CSV
using DataFrames

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    output = joinpath(
        root, "results", "data_audit",
        "legacy_dependencies.csv"
    )
    mkpath(dirname(output))

    legacy = [
        "population", "fertility", "mortality",
        "migration", "contact",
        "tb_notifications_by_age.csv",
        "tb_notifications_total.csv",
        "tb_recurrent_notifications.csv",
        "tb_notifications_by_site.csv",
        "tb_treatment_outcomes_main.csv",
        "tb_historical_new_notifications_by_age_and_type.csv",
        "tb_incidence.csv",
    ]

    report = DataFrame(
        legacy_file=String[],
        consumer=String[],
        line=Int[],
        code=String[],
    )

    # Search source, scripts, examples and tests.
    for directory in ("src", "scripts", "examples", "test")
        base = joinpath(root, directory)
        isdir(base) || continue

        for (dir, _, files) in walkdir(base)
            for filename in files
                endswith(filename, ".jl") || continue

                path = joinpath(dir, filename)
                relative = replace(relpath(path, root), '\\' => '/')

                for (number, line) in enumerate(eachline(path))
                    for name in legacy
                        occursin("\"$name\"", line) ||
                            occursin("\"$name\"", replace(line, "'" => "\"")) ||
                            continue

                        push!(report, (
                            name, relative, number, strip(line)
                        ))
                    end
                end
            end
        end
    end

    sort!(report, [:legacy_file, :consumer, :line])
    unique!(report)

    CSV.write(output, report)
    println("Found ", nrow(report), " references")
    println("Report: ", output)
end
