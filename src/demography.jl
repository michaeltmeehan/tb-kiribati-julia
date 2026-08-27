using DelimitedFiles

const POPULATION = readdlm(".\\data\\population", Float64)
const MIN_POP_YEAR = 1950.
const MAX_POP_YEAR = 2100.


const FERTILITY = readdlm(".\\data\\fertility", Float64)
const MIN_FERT_AGE = 10
const MAX_FERT_AGE = 54
const FEMALE_FRACTION = 0.485

const MORTALITY = readdlm(".\\data\\mortality", Float64)
const MIN_MORT_AGE = 0
const MAX_MORT_AGE = 100


const MIGRATION = readdlm(".\\data\\migration", Float64)
const MIN_MIGRATION_AGE = 0
const MAX_MIGRATION_AGE = 100


# TODO: Check whether I want the fertility and mortality rates time clamped - I probably just want them zeroed beyond the time window
# All datasets begin in the year 1950 and run up to 2100
@inline function year_index(t::Float64)
    t = clamp(t, MIN_POP_YEAR, MAX_POP_YEAR)
    Int(floor(t) - MIN_POP_YEAR) + 1
end

# Fertility rates begin at age 10 and go up to age 54
@inline function fertility_rate(a::Int, t::Float64)
    (a < MIN_FERT_AGE || a > MAX_FERT_AGE) && return 0.
    FEMALE_FRACTION * FERTILITY[a - MIN_FERT_AGE + 1, year_index(t)]
end

# Mortality rates begin at age 0 and go up to age 100
@inline function mortality_rate(a::Int, t::Float64)
    a = clamp(a, MIN_MORT_AGE, MAX_MORT_AGE)
    MORTALITY[a + 1, year_index(t)]
end

@inline function migration_flow(t::Float64)
    return MIGRATION[year_index(t)] * 1e3
end

  
function get_population(t::Float64)
    population = zeros(NAGE)
    population[1:(NAGE-1)] .= POPULATION[1:(NAGE-1), year_index(t)]
    population[end] = sum(POPULATION[(NAGE:end), year_index(t)])
    return population * 1e3
end


function _apply_demography!(u, t)
    newborns = 0.0
    total_pop = 0.0
    # Apply births and deaths
    for a in 1:NAGE
        base = (a - 1) * NSTATE

        fertility = fertility_rate(a - 1, t)
        mortality = mortality_rate(a - 1, t)

        for c in 1:NEPI
            x = u[base + c]

            # Contribution to newborn population
            newborns += fertility * x

            # Remove deaths and emigrants
            u[base + c] = (1 - mortality) * x

            total_pop += u[base + c]
        end
    end

    # Apply migration
    net_migration = migration_flow(t)
    for a in 1:NAGE
        base = (a - 1) * NSTATE
        for c in 1:NEPI
            u[base + c] += u[base + c] * net_migration / total_pop
        end
    end

    # Apply ageing
    # Oldest age band (95+) retains its population and receives age 94
    for c in 1:NEPI
        u[(NAGE - 1) * NSTATE + c] +=
            u[(NAGE - 2) * NSTATE + c]
    end

    # Ages 1--94 receive the immediately younger cohort
    for a in (NAGE - 1):-1:2
        younger = (a - 2) * NSTATE
        older   = (a - 1) * NSTATE

        for c in 1:NEPI
            u[older + c] = u[younger + c]
        end
    end

    # Clear age-0 epi states and introduce newborns as Mtb naive
    for c in 1:NEPI
        u[c] = 0.0
    end
    u[MtbNaive] = newborns

    return nothing
end

# TODO: Do not apply demography during burn-in period (i.e., integrator.t < 1950 && return)
function apply_demography!(integrator)
    if integrator.t <= STATIC_YEAR
        _apply_static_demography!(integrator.u)
    else
        _apply_demography!(integrator.u, integrator.t - 1.0)
    end
end


function get_age_distribution(u)
    pop = zeros(NAGE)
    for a in 1:NAGE
        total = 0.
        base = (a - 1) * NSTATE
        for c in 1:NEPI
            total += u[base + c]
        end
        pop[a] = total
    end
    return pop
end