using DelimitedFiles

const POPULATION = readdlm(".\\data\\population", Float64)
const MIN_POP_YEAR = 1950.
const MAX_POP_YEAR = 2100.


const FERTILITY = readdlm(".\\data\\fertility", Float64)
const MIN_FERT_AGE = 10
const MAX_FERT_AGE = 54

const MORTALITY = readdlm(".\\data\\mortality", Float64)
const MIN_MORT_AGE = 0
const MAX_MORT_AGE = 100

# All datasets begin in the year 1950 and run up to 2100
@inline function year_index(t::Float64)
    t = clamp(t, MIN_POP_YEAR, MAX_POP_YEAR)
    Int(round(t) - 1950) + 1
end

# Fertility rates begin at age 10 and go up to age 54
@inline function fertility_rate(a::Int, t::Float64)
    a < MIN_FERT_AGE || a > MAX_FERT_AGE && return 0.
    FERTILITY[a - 10 + 1, year_index(t)]
end

# Mortality rates begin at age 0 and go up to age 100
@inline function mortality_rate(a::Int, t::Float64)
    a = clamp(a, MIN_MORT_AGE, MAX_MORT_AGE)
    MORTALITY[a + 1, year_index(t)]
end

@inline migration_inflow(a, t) =
    0.
    # MIGRATION_IN[a, year_index(t)]

@inline migration_outflow(a, t) =
    0.
    # MIGRATION_OUT[a, year_index(t)]

  
function get_population(t::Float64)
    population = zeros(NAGE)
    population[1:(NAGE-1)] .= POPULATION[1:(NAGE-1), year_index(t)]
    population[end] = sum(POPULATION[(NAGE:end), year_index(t)])
    return population * 1e3
end


function _apply_demography!(u, t)
    newborns = 0.0

    # Apply births, deaths and migration outflow
    for a in 1:NAGE
        base = (a - 1) * NSTATE

        fertility = fertility_rate(a, t)
        mortality = mortality_rate(a, t)
        outflow = migration_outflow(a, t)

        for c in 1:NEPI
            x = u[base + c]

            # Contribution to newborn population
            newborns += fertility * x

            # Remove deaths and emigrants
            u[base + c] = (1 - mortality - outflow) * x
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
    for c in 2:NEPI
        u[c] = 0.0
    end
    u[MtbNaive] = newborns

    # Migration inflow into the new age distribution
    for a in 1:NAGE
        base = (a - 1) * NSTATE
        u[base + MtbNaive] += migration_inflow(a, t)
    end

    return nothing
end

function apply_demography!(integrator)
    _apply_demography!(integrator.u, integrator.t)
end