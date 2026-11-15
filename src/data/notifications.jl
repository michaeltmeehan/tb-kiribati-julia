using CSV

"""
    load_notification_observations(path; age_labels, years, aggregate_labels=Dict("0-14" => ["0-4", "5-14"]))

Read `year,age_group,cases`. Each annual observation stores a list of model
age-bin indices for each reported count. Prefer disaggregated counts when
complete; otherwise use a reported aggregate. Never double-count both.
"""
function load_notification_observations(path::AbstractString;
    age_labels, years, aggregate_labels=Dict("0-14" => ["0-4", "5-14"]))
    labels = String.(age_labels)
    length(unique(labels)) == length(labels) || throw(ArgumentError("Duplicate model age labels"))
    aggregates = Dict(String(k) => String.(v) for (k, v) in pairs(aggregate_labels))
    all(all(in(labels), v) for v in values(aggregates)) ||
        throw(ArgumentError("Aggregate labels must refer to model age groups"))
    rows = collect(CSV.File(path))
    result = NamedTuple[]
    for year in years
        values_by_label = Dict{String,Int}()
        for row in rows
            row.year == year || continue
            label = String(row.age_group)
            (label in labels || haskey(aggregates, label)) ||
                throw(ArgumentError("Unknown age group $label in $year"))
            ismissing(row.cases) && continue
            haskey(values_by_label, label) && throw(ArgumentError("Duplicate $label in $year"))
            y = Int(row.cases)
            y >= 0 || throw(ArgumentError("Negative count in $year"))
            values_by_label[label] = y
        end
        isempty(values_by_label) && throw(ArgumentError("No counts for $year"))
        # Check redundant aggregate totals whenever all constituents are present.
        for (name, constituents) in aggregates
            if haskey(values_by_label, name) && all(haskey(values_by_label, c) for c in constituents)
                values_by_label[name] == sum(values_by_label[c] for c in constituents) ||
                    throw(ArgumentError("Inconsistent aggregate $name in $year"))
            end
        end
        observed_labels = String[]
        observed_counts = Int[]
        observed_groups = Vector{Int}[]
        used = Set{String}()
        for (i, label) in enumerate(labels)
            label in used && continue
            if haskey(values_by_label, label)
                push!(observed_labels, label)
                push!(observed_counts, values_by_label[label])
                push!(observed_groups, [i])
                push!(used, label)
            else
                matching = [k for (k, v) in aggregates if label in v && haskey(values_by_label, k)]
                length(matching) == 1 || throw(ArgumentError("Missing/ambiguous $label in $year"))
                name = only(matching)
                constituents = aggregates[name]
                # Accept aggregate only if none of its members have already been used.
                all(c -> !(c in used), constituents) ||
                    throw(ArgumentError("Partially reported aggregate $name in $year"))
                push!(observed_labels, name)
                push!(observed_counts, values_by_label[name])
                push!(observed_groups, [findfirst(==(c), labels) for c in constituents])
                union!(used, constituents)
            end
        end
        push!(result, (year=Int(year), labels=observed_labels,
            counts=observed_counts, groups=observed_groups))
    end
    return result
end
