using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using TBKiribatiJulia

sol = simulate_demo()
println("retcode = ", sol.retcode)
println("saved steps = ", length(sol.t))

final_state = reshape(sol.u[end], NSTATE, NAGE)
epi_total = sum(final_state[1:NEPI, :])
cum_total = sum(final_state[NEPI + 1:end, :])

println("final epidemiological total = ", epi_total)
println("final cumulative total = ", cum_total)
