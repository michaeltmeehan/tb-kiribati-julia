
using CSV
using DataFrames
using SHA
using Dates

let
    root = normpath(joinpath(@__DIR__, "..", ".."))
    metadata_dir = joinpath(root, "data", "metadata")
    mkpath(metadata_dir)

    # Inventory data and the code that generates it.
    locations = (
        ("raw", "data/raw"),
        ("processed", "data/processed"),
        ("processing_code", "src/data"),
        ("processing_code", "src/mixing"),
        ("pipeline", "scripts/data"),
    )

    # Windows-independent relative paths.
    relative(path) = replace(relpath(path, root), '\\' => '/')

    function file_hash(path)
        open(path, "r") do io
            return bytes2hex(sha256(io))
        end
    end

    manifest = DataFrame(
        category=String[],
        path=String[],
        size_bytes=Int64[],
        sha256=String[],
    )

    for (category, location) in locations
        directory = joinpath(root, location)
        isdir(directory) || continue

        for (dir, _, files) in walkdir(directory)
            for filename in files
                path = joinpath(dir, filename)

                # Only actual files, excluding editor artefacts.
                isfile(path) || continue
                startswith(filename, ".") && continue
                endswith(filename, "~") && continue

                push!(manifest, (
                    category,
                    relative(path),
                    filesize(path),
                    file_hash(path),
                ))
            end
        end
    end

    sort!(manifest, [:category, :path])

    # Prevent accidentally generating an empty manifest.
    for category in ("raw", "processed")
        any(manifest.category .== category) ||
            error("No $category files found")
    end

    output = joinpath(metadata_dir, "file_manifest.csv")
    CSV.write(output, manifest)

    println("\nDATA PROVENANCE MANIFEST")
    println(combine(
        groupby(manifest, :category),
        nrow => :files,
        :size_bytes => sum => :total_bytes,
    ))

    println("\nManifest written to: ", output)
    println("Recorded ", nrow(manifest), " files")
end
