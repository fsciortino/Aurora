using PyCall, aurora, BenchmarkTools#, StaticArrays, SpecialFunctions

cd(@__DIR__)
pushfirst!(PyVector(pyimport("sys")."path"), "")
test = pyimport("fake_test")


nt_out, t_trans, D, V, dv_t, sint_t, s_t,
             al_t, rr, pro, qpr, r_saw, dlen, time, saw, flx, it_out, dsaw, rcl,
             divbls, taudiv, taupump, tauwret, rvol_lcfs, dbound, dlim, prox,
             rn_t0 = test.run()

@btime aurora.run(nt_out, t_trans, D, V, dv_t, sint_t, s_t,
             al_t, rr, pro, qpr, r_saw, dlen, time, saw, flx, it_out, dsaw, rcl,
             divbls, taudiv, taupump, tauwret, rvol_lcfs, dbound, dlim, prox,
             rn_t0)

nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
test.plotJuliaResults(out)
print(rclw_rate)
