const EQUILIBRIUM_YEAR = 2025.0
const EQUILIBRIUM_POPULATION = get_population(EQUILIBRIUM_YEAR)

const EQUILIBRIUM_MORTALITY = [mortality_rate(a - 1, EQUILIBRIUM_YEAR) for a in 1 :NAGE]

const EQUILIBRIUM_SURVIVAL = exp.(-EQUILIBRIUM_MORTALITY)

const EQUILIBRIUM_NEWBORNS = EQUILIBRIUM_POPULATION[1]

const EQUILIBRIUM_SURVIVORS = EQUILIBRIUM_SURVIVAL .* EQUILIBRIUM_POPULATION

const EQUILIBRIUM_MIGRATION = let 
    net_migration = zeros(NAGE)

    # Age 0 is supplied by births (no migration balancing required)
    
    for a in 2:(NAGE - 1)
        net_migration[a] = EQUILIBRIUM_POPULATION[a] - EQUILIBRIUM_SURVIVORS[a - 1]
    end

    net_migration[end] = EQUILIBRIUM_POPULATION[end] - EQUILIBRIUM_SURVIVORS[end] - EQUILIBRIUM_SURVIVORS[end - 1]

    net_migration
    
end


function _apply_equilibrium_demography!(u)

    for a in 1:NAGE
        base = (a - 1) * NSTATE
        # survival = 1.0 - EQUILIBRIUM_MORTALITY[a]
        survival = EQUILIBRIUM_SURVIVAL[a]

        for c in 1:NEPI
            u[base + c] *= survival
        end
    end

    # Apply ageing (starting with oldest age group)
    for c in 1:NEPI
        u[(NAGE - 1) * NSTATE + c] += u[(NAGE - 2) * NSTATE + c]
    end

    for a in (NAGE - 1):-1:2
        for c in 1:NEPI
            u[(a - 1) * NSTATE + c] = u[(a - 2) * NSTATE + c]
        end
    end

    # Clear age 0
    for c in 1:NEPI
        u[c] = 0.0
    end

    for a in 2:NAGE
        base = (a - 1) * NSTATE

        age_pop = 0.0
        for c in 1:NEPI
            age_pop += u[base + c]
        end

        net_migration = EQUILIBRIUM_MIGRATION[a]

        if age_pop > 0.0
            scale = 1.0 + net_migration / age_pop

            for c in 1:NEPI
                u[base + c] *= scale
            end

        elseif net_migration > 0.0
            u[base + MtbNaive] = net_migration
        end
    end

    # Introduce newborn
    u[MtbNaive] = EQUILIBRIUM_NEWBORNS

    _reconcile_population!(
    u,
    EQUILIBRIUM_POPULATION,
)
    
    return
end


function _reconcile_population!(
    u,
    target_population::AbstractVector{<:Real},
)
    length(target_population) == NAGE ||
        error("target population vector must have length $NAGE")

    for a in 1:NAGE
        base = (a - 1) * NSTATE

        age_pop = 0.0
        for c in 1:NEPI
            age_pop += u[base + c]
        end

        target = Float64(target_population[a])

        if age_pop > 0.0
            scale = target / age_pop

            for c in 1:NEPI
                u[base + c] *= scale
            end

        elseif target > 0.0
            # Defensive fallback if an age group has somehow become empty.
            u[base + MtbNaive] = target
        end
    end

    return nothing
end