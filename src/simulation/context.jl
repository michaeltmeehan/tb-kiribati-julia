struct TBWorkspace
    foi::Vector{Float64}
    infectious_prevalence::Vector{Float64}
end

TBWorkspace(nages::Int) =
    TBWorkspace(zeros(nages), zeros(nages))

struct SimulationContext{M<:AbstractTBModel,P,D}
    model::M
    params::P
    demography::D
    workspace::TBWorkspace
end