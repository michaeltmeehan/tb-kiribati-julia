const STATIC_YEAR = 2025.0
const STATIC_POPULATION = get_population(STATIC_YEAR)

const STATIC_MORTALITY = [mortality_rate(a - 1, STATIC_YEAR) for a in 1 :NAGE]

const STATIC_NEWBORNS = STATIC_POPULATION[1]

const STATIC_SURVIVORS = (1. .- STATIC_MORTALITY) .* STATIC_POPULATION

const STATIC_MIGRATION = let 
    net_migration = zeros(NAGE)

    # Age 0 is supplied by births (no migration balancing required)
    
    for a in 2:(NAGE - 1)
        net_migration[a] = STATIC_POPULATION[a] - STATIC_SURVIVORS[a - 1]
    end

    net_migration[end] = STATIC_POPULATION[end] - STATIC_SURVIVORS[end] - STATIC_SURVIVORS[end - 1]

    net_migration
    
end


function _apply_static_demography!(u)

    for a in 1:NAGE
        base = (a - 1) * NSTATE
        survival = 1.0 - STATIC_MORTALITY[a]

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

        net_migration = STATIC_MIGRATION[a]

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
    u[MtbNaive] = STATIC_NEWBORNS
    
    return
end
