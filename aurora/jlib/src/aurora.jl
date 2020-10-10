module aurora
@inline function run(nt_out, t_trans, D, V, dv_t, sint_t, s_t,
             al_t, rr, pro, qpr, r_saw, dlen, time, saw, flx, it_out, dsaw, rcl,
             divbls, taudiv, taupump, tauwret, rvol_lcfs, dbound, dlim, prox,
             rn)
    """
    !
    !     *******************************************************************
    !     * solve the coupled impurity transport equations and return
    !     * the impurity densities at all requested times and radial points
    !     *******************************************************************
    !
    !     ------ All inputs are in CGS units -----
    !
    !   Get list of required input parameters in Python using
    !        print(aurora.run.__doc__)
    !
    !     INPUTS:
    !     * nion      integer
    !                 number of ionization stages
    !     * ir        integer
    !                 number of radial grid points
    !     * nt        integer
    !                 number of time steps for the solution of the transport eq.
    !     * nt_out    integer
    !                 number of times at which the impurity densities shall
    !                 be saved
    !     * nt_trans  integer
    !                 number of times at which D,V profiles are given.
    !     * t_trans   real*8 (nt_trans)
    !                 times at which transport coefficients change [s].
    !     * D         real*8 (ir,nt_trans,nion)
    !                 diffusion coefficient on time and radial grids [cm^2/s]
    !                 This must be given for each charge state and time.
    !     * V         real*8 (ir,nt_trans,nion)
    !                 drift velocity on time and radial grids [cm/s]
    !                 This must be given for each charge state and time.
    !     * dv_t      real*8 (ir,nt)
    !                 frequency for parallel loss on radial and time grids [1/s]
    !     * sint_t    real*8 (ir,nt)
    !                 Radial profile of neutrals over time: n0(ir,t) = flx*sint(ir,t).
    !     * s_t       real*8 (ir,nion,nt)
    !                 ionisation rates (nz=nion must be filled with zeros).
    !                 Note that this is time dependent!
    !     * al_t      real*8 (ir,nion,nt)
    !                 recombination rates (nz=nion must be filled with zeros)
    !                 Note that this is time dependent!
    !     * rr        real*8 (ir)
    !                 the radial coordinate r (=rho volume)
    !     * pro       real*8 (ir)
    !                 the radial mesh is equidistant in the coordinate rho
    !                 with step size d_rho:
    !                 pro = (drho/dr)/(2 d_rho) = rho'/(2 d_rho)
    !     * qpr       real*8 (ir)
    !                 the radial mesh is equidistant in the coordinate rho
    !                 with step size d_rho:
    !                 qpr = (d^2 rho/dr^2)/(2 d_rho) = rho''/(2 d_rho)
    !     * r_saw     real*8
    !                 inversion radius for sawteeth [cm]
    !     * dlen      real*8
    !                 decay length at last radial grid point
    !     * time      real*8 (nt)
    !                 time grid for transport solver
    !     * saw       integer (nt)
    !                 switch to induce a sawtooth crashes
    !                 if saw(it) eq 1 there is a crash at time(it)
    !     * flx       real*8 (nt)
    !                 impurity influx per unit length and time [1/cm/s]
    !                 (only 1 to nt-1 are used)
    !     * it_out    integer (nt)
    !                 store the impurity distributions if it_out(it).eq.1
    !     * dsaw      real*8
    !                 Width of sawtooth crash region.
    !     * rcl       real*8
    !                 Wall recycling coefficient. Normally, this would be in the range [0,1].
    !                 However, if set to a value <0, then this is interpreted as a flag, indicating
    !                 that particles in the divertor should NEVER return to the main plasma.
    !                 This is effectively what the rclswitch flag does in STRAHL (kind of confusingly).
    !     * divbls    real*8
    !                 Divertor puff rate, given as a fraction of the total flux given by flx.
    !                 This is currently only used in the absence of recycling!
    !     * taudiv    real*8
    !                 Time scale for transport out of the divertor reservoir
    !     * taupump   real*8
    !                 Time scale for impurity elimination through out-pumping
    !     * tauwret   real*8
    !                 Time scale of temporary retention at the wall
    !     * rvol_lcfs real*8
    !                 Radius (in r_vol units) at which the LCFS is located (from grid file)
    !     * dbound    real*8
    !                 Width of the SOL, given by r_bound - r_lcfs (from param file, cm in r_vol).
    !                 This value sets the width of the radial grid.
    !     * dlim      real*8
    !                 Position of the limiter wrt to the LCFS, i.e. r_lim - r_lcfs (cm, in r_vol units).
    !                 Inside of this limiter location, the parallel connection length to the divertor applies,
    !                 while outside of it the relevant connection length is the one to the limiter.
    !                 NB: these different connection lengths must be taken into consideration when
    !                 preparing the dv (frequency for parallel loss on radial grid).
    !     * prox      real*8
    !                 Grid parameter for loss rate at the last radial point, returned by `get_radial_grid' subroutine.
    !     * rn        real*8 (ir,nion), optional
    !                 Impurity densities at the start time [1/cm^3]. If not provided, all elements are
    !                 set to 0.
    !
    ! **************************
    !     OUTPUTS:
    !
    !     * rn_out    real*8 (ir,nion,nt_out)
    !                 Impurity densities (temporarily) in the magnetically-confined plasma at the requested times [1/cm^3].
    !     * N_ret     real*8 (nt_out)
    !                 Impurity densities (permanently) retained at the wall over time [1/cm^3].
    !     * N_wall    real*8 (nt_out)
    !                 Impurity densities (temporarily) at the wall over time [1/cm^3].
    !     * N_div     real*8 (nt_out)
    !                 Impurity densities (temporarily) in the divertor reservoir over time [1/cm^3].
    !     * N_pump    real*8 (nt_out)
    !                 Impurity densities (permanently) in the pump [1/cm^3].
    !     * N_tsu     real*8 (nt_out)
    !                 Edge loss [1/cm^3].
    !     * N_dsu     real*8 (nt_out)
    !                 Parallel loss [1/cm^3].
    !     * N_dsul    real*8 (nt_out)
    !                 Parallel loss to limiter [1/cm^3].
    !     * rcld_rate real*8 (nt_out)
    !                 Recycling from divertor [1/cm^3/s].
    !     * rclw_rate real*8 (nt_out)
    !                 Recycling from wall [1/cm^3/s].
    !
    !     *******************************************************************

    """
    # Sizing info
    ir, nt_trans, nion = size(D)
    _, nt = size(dv_t)

    # Intermediaries
    ra = Array{Float64,2}(undef, ir, nion)
    diff = Array{Float64,2}(undef, ir, nion)
    conv = Array{Float64,2}(undef, ir, nion)

    # For use in impden!
    a = Array{Float64,2}(undef, ir, nion)
    b = Array{Float64,2}(undef, ir, nion)
    c = Array{Float64,2}(undef, ir, nion)
    d1 = Vector{Float64}(undef, ir)
    bet = Vector{Float64}(undef, ir)
    gam = Vector{Float64}(undef, ir)

    # Outputs
    rn_out = Array{Float64,3}(undef, ir, nion, nt_out)
    N_wall = Vector{Float64}(undef, nt_out)
    N_div = Vector{Float64}(undef, nt_out)
    N_pump = Vector{Float64}(undef, nt_out)
    N_ret = Vector{Float64}(undef, nt_out)
    N_tsu = Vector{Float64}(undef, nt_out)
    N_dsu = Vector{Float64}(undef, nt_out)
    N_dsul = Vector{Float64}(undef, nt_out)
    rcld_rate = Vector{Float64}(undef, nt_out)
    rclw_rate = Vector{Float64}(undef, nt_out)

    rcld = 0.
    rclw = 0.
    Nret = 0.
    tve = 0.
    divnew = 0.
    npump = 0.
    tsu = 0.
    dsu = 0.
    dsul = 0.

    it = 1
    kt = 1
    # Output if indicated
    @inbounds if it_out[it] == 1
        rn_out[:,:,kt] = rn

        N_wall[kt] = tve
        N_div[kt] = divnew
        N_pump[kt] = npump
        N_tsu[kt] = tsu
        N_dsu[kt] = dsu
        N_dsul[kt] = dsul

        N_ret[kt] = Nret
        rcld_rate[kt] = 0.
        rclw_rate[kt] = 0.
        kt += 1
    end

    # ============ time loop: ==========
    @inbounds for it in 2:nt
        dt = time[it] - time[it-1]

        ra .= rn # update old array to new (from previous time step)

        for nz in 1:nion
            # updated transport coefficients for each charge state
            linip_arr!(nt_trans, ir, t_trans, view(D, :, :, nz), time[it], view(diff, :, nz))
            linip_arr!(nt_trans, ir, t_trans, view(V, :, :, nz), time[it], view(conv, :, nz))
        end

        divold = divnew # strahl.f, L756

        # pick current ioniz and recomb coeffs. || loss rate and source radial prof
        al = view(al_t,:,:,it)
        s = view(s_t,:,:,it)
        dv = view(dv_t,:,it)
        sint = view(sint_t,:,it)

        # evolve impurity density with current transport coeffs
        Nret, rcld, rclw = impden!(nion, ir, ra, rn, diff, conv, dv, sint, s, al, rr,
                        pro, qpr, flx[it-1], dlen, dt, rcl, tsu, dsul, divold,
                        divbls, taudiv, tauwret, Nret, rcld, rclw, a, b, c, d1, bet, gam)

        # sawteeth
        if saw[it] == 1
            saw_mix!(nion, ir, rn, r_saw, dsaw, rr, pro)
        end

        # particle losses at wall & divertor + recycling
        tve, divnew, npump, tsu, dsu, dsul = edge_model!(nion, ir, ra, rn, diff, conv, dv, dt, rvol_lcfs,
                            dbound, dlim, prox, rr, pro, rcl, taudiv, taupump,
                            divbls, divold, flx[it-1], divnew, tve, npump, tsu,
                            dsu, dsul)

        # array time-step saving/output
        if it_out[it] == 1
            rn_out[:,:,kt] = rn

            N_wall[kt] = tve
            N_div[kt] = divnew
            N_pump[kt] = npump
            N_tsu[kt] = tsu
            N_dsu[kt] = dsu
            N_dsul[kt] = dsul

            N_ret[kt] = Nret
            rcld_rate[kt] = rcld
            rclw_rate[kt] = rclw
            kt += 1
        end
    end

    # End of time loop

    return rn_out, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate
end

@inline function impden!(nion, ir, ra, rn, diff, conv, dv, sint, s, al, rr, pro, qpr,
                flx, dlen, det, rcl, tsuold, dsulold, divold, divbls, taudiv,
                tauwret, Nret, rcld, rclw, a, b, c, d1, bet, gam)
    """

    """

    # dr near magnetic axis

    der = rr[2] - rr[1]

    #     **************************************************
    #     ** Time CENTERED TREATMENT in radial coordinate **
    #     ** LACKNER METHOD IN Z                          **
    #     ** time step is split in two half time steps    **
    #     **************************************************
    flxtot = flx

    # ------ Recycling ------  ! impden.f  L328-L345
    # Part of the particles that hit the wall are taken to be fully-stuck. Another part is only temporarily retained at the wall.
    # Particles FULLY STUCK to the wall will never leave, i.e. tve can only increase over time (see above).
    # Particles that are only temporarily retained at the wall (given by the recycling fraction) come back
    # (recycle) according to the tauwret time scale.

    if rcl >= 0. # activated divertor return (R>=0) + recycling mode (if R>0)
        rcld = divold/taudiv
        rclw = rcl*(tsuold+dsulold)

        if tauwret > 0.
            # include flux from particles previously retained at the wall
            Nret = Nret * (1 - det/tauwret) + rclw*det # number of particles temporarily retained at wall
            rclw = Nret/tauwret # component that goes back to be a source
        end

        flxtot = flx*(1-divbls) + rclw + rcld
    else # no divertor return at all
        rcld = 0.
        rclw = 0.
    end

    @inbounds for i in 1:ir
        rn[i,1] = flxtot*sint[i] # index of 1 stands for first ionization stage
    end

    dt = det/2.

    ########### First half time step direction up #########

    @inbounds for nz in 2:nion
        #   r = 0
        a[1,nz] = 0.
        c[1,nz] = -2.0*dt*diff[1,nz]*pro[1]
        b[1,nz] = 1.0 - c[1,nz] + 2.0*dt*conv[2,nz]/der
        d1[1] = ra[1,nz]*(2.0-b[1,nz])-ra[2,nz]*c[1,nz]

        #   r=r or at r+db respectively

        if dlen > 0.
            temp1 = 4.0*dt*pro[ir]^2*diff[ir,nz]
            temp2 = 0.5*dt*(qpr[ir]*diff[ir,nz]-pro[ir]*conv[ir,nz])
            temp3 = 0.5*dt*(dv[ir] + conv[ir,nz]/rr[ir])
            temp4 = 1.0/pro[ir]/dlen

            a[ir,nz] = -temp1
            b[ir,nz] = 1.0+(1.0+0.5*temp4)*temp1 + temp4*temp2 + temp3
            c[ir,nz] = 0.
            d1[ir] = -ra[ir-1,nz]*a[ir,nz] + ra[ir,nz]*(2.0 - b[ir,nz])
            b[ir,nz] += dt*s[ir,nz]
            d1[ir] -= dt*(ra[ir,nz]*al[ir,nz-1] - rn[ir,nz-1]*s[ir,nz-1])

            if nz < nion
                d1[ir] += dt*ra[ir,nz+1]*al[ir,nz]
            end
        else # Edge Conditions for rn[ir]=0
            a[ir,nz] = 0
            b[ir,nz] = 1.
            c[ir,nz] = 0.
            d1[ir] = ra[ir,nz]
        end

        #   normal coefficients

        for i in 2:ir-1
            temp1 = dt*pro[i]^2
            temp2 = 4.0*temp1*diff[i,nz]
            temp3 = dt/2.0*(dv[i] + pro[i]*(conv[i+1,nz] - conv[i-1, nz]) + conv[i,nz]/rr[i])
            a[i,nz] = (0.5*dt*qpr[i]*diff[i,nz] + temp1*(0.5*(diff[i+1,nz] -
                     diff[i-1,nz] - conv[i,nz]/pro[i]) - 2.0*diff[i,nz]))
            b[i,nz] = 1.0 + temp2 + temp3
            c[i,nz] = -temp2 - a[i,nz]
            d1[i] = -ra[i-1,nz]*a[i,nz] + ra[i,nz]*(2.0-b[i,nz]) - ra[i+1,nz]*c[i,nz]
        end

        for i in 1:ir-1
            b[i,nz] += dt*s[i,nz]
            d1[i] -= dt*(ra[i,nz]*al[i,nz-1] - rn[i,nz-1]*s[i,nz-1]) # ra --> "alt" (i-1) density; rnz --> "current" (i) density

            if nz < nion
                d1[i] += dt*ra[i,nz+1]*al[i,nz] # al --> recomb; s --> ioniz
            end
        end

        #       solution of tridiagonal equation system
        bet[1] = b[1,nz]
        gam[1] = d1[1]/b[1,nz]
        for i in 2:ir
            bet[i] = b[i,nz] - (a[i,nz]*c[i-1,nz])/bet[i-1]
            gam[i] = (d1[i] - a[i,nz]*gam[i-1])/bet[i]
        end
        rn[ir,nz] = gam[ir]

        for i in ir-1:-1:1
            rn[i,nz] = gam[i] - (c[i,nz]*rn[i+1,nz])/bet[i]
        end
    end

    ########### Second half time step direction down #########

    @inbounds for nz in nion:-1:2
        #   r = 0
        a[1,nz] = 0.
        c[1,nz] = -2.0*dt*diff[1,nz]*pro[1]
        b[1,nz] = 1.0 - c[1,nz] + 2.0*dt*conv[2,nz]/der
        d1[1] = rn[1,nz]*(2.0 - b[1,nz]) - rn[2,nz]*c[1,nz]

        #       r=r or at r+db respectively

        if dlen > 0.
            temp1 = 4.0*dt*pro[ir]^2*diff[ir,nz]
            temp2 = 0.5*dt*(qpr[ir]*diff[ir,nz] - pro[ir]*conv[ir,nz])
            temp3 = 0.5*dt*(dv[ir] + conv[ir,nz]/rr[ir])
            temp4 = 1.0/pro[ir]/dlen

            a[ir,nz] = -temp1
            b[ir,nz] = 1.0 + (1.0 + 0.5*temp4)*temp1 + temp4*temp2 + temp3
            c[ir,nz] = 0.
            d1[ir] = -rn[ir-1,nz]*a[ir,nz] + rn[ir,nz]*(2.0 - b[ir,nz])
            b[ir,nz] += dt*al[ir,nz-1]
            d1[ir] -= dt*(rn[ir,nz]*s[ir,nz] - rn[ir,nz-1]*s[ir,nz-1])

            if nz < nion
                d1[ir] += dt*rn[ir,nz+1]*al[ir,nz]
            end
        else # Edge conditions for rn[ir]=0.
            a[ir,nz] = 0
            b[ir,nz] = 1.
            c[ir,nz] = 0.
            d1[ir] = rn[ir,nz]
        end

        #       normal coefficients

        for i in 2:ir-1
            temp1 = dt*pro[i]^2
            temp2 = 4.0*temp1*diff[i,nz]
            temp3 = dt/2.0*(dv[i] + pro[i]*(conv[i+1,nz] - conv[i-1,nz]) + conv[i,nz]/rr[i])

            a[i,nz] = (0.5*dt*qpr[i]*diff[i,nz] + temp1*(0.5*(diff[i+1,nz]-
                      diff[i-1,nz] - conv[i,nz]/pro[i]) - 2.0*diff[i,nz]))
            b[i,nz] = 1.0 + temp2 + temp3
            c[i,nz] = -temp2 - a[i,nz]
            d1[i] = -rn[i-1,nz]*a[i,nz] + rn[i,nz]*(2.0 - b[i,nz]) - rn[i+1,nz]*c[i,nz]
        end

        for i in 1:ir-1
            b[i,nz] += dt*al[i,nz-1]
            d1[i] -= dt*(rn[i,nz]*s[i,nz] - rn[i,nz-1]*s[i,nz-1])

            if nz < nion
                d1[i] += dt*rn[i,nz+1]*al[i,nz]*(nz < nion)
            end
        end

        #   solution of tridiagonal equation system

        bet[1] = b[1,nz]
        gam[1] = d1[1]/b[1,nz]
        for i in 2:ir
            bet[i] = b[i,nz] - (a[i,nz]*c[i-1,nz])/bet[i-1]
            gam[i] = (d1[i] - a[i,nz]*gam[i-1])/bet[i]
        end
        rn[ir,nz] = gam[ir]

        for i in ir-1:-1:1
            rn[i,nz] = gam[i] - (c[i,nz]*rn[i+1,nz])/bet[i]
        end
    end
    return Nret, rcld, rclw
end

@inline function saw_mix!(nion, ir, rn, rsaw, dsaw, rr, pro)
    """

    """
    #   index of mixing radius
    @inbounds for imix in 1:ir
        if rr[i] > rsaw
            break
        end
    end

    @inbounds for nz in 2:nion    # loop over ionized stages

        #   area integral in mixing radius of old profile

        sum_old = (0.125*(rn[imix,nz]*rr[imix]/pro[imix] -
                   rn[imix-1,nz]*rr[imix-1]/pro[imix-1])) # only use new density, rn
        for i in 2:imix-1
            sum_old += rn[i,nz]*rr[i]/pro[i]
        end

        #   ERFC sawtooth crash model
        ff = sum_old/rr[imix]^2 # nmean
        for i in 1:ir
            rn[i,nz] = (ff/2.0*erfc((rr[i] - rsaw)/dsaw) + (rn[i,nz]/2.0)*
                        erfc(-(rr[i]-rsaw)/dsaw))
        end

        #      flat profile
        #  ff = sum_old/rr(imix)**2
        #  do i=1,imix-1
        #    rn(i,nz) = ff
        #  end do
        #  rn(imix,nz) = (ra(imix+1,nz)+ff)/2.

        #      area integral in mixing radius of new profile
        sum = (0.125*(rn[imix,nz]*rr[imix]/pro[imix] -
               rn[imix-1,nz]*rr[imix-1]/pro[imix-1]))
        for i in 2:imix-1
            sum += rn[i,nz]*rr[i]/pro[i]
        end

        #   ensure particle conservation

        ff = sum_old/sum
        for i in 1:imix
            rn[i,nz] = rn[i,nz]*ff
        end
    end
end

@inline function edge_model!(nion, ir, ra, rn, diff, conv, dv, det, rvol_lcfs, dbound,
                    dlim, prox, rr, pro, rcl, taudiv, taupump, divbls, divold,
                    flx, divnew, tve, npump, tsu, dsu, dsul)
    """

    """
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #      Compute edge fluxes given multi-reservoir parameters
    #   Core-densities do not directly depend on this -- but recycling can only be activated
    #   if this 1D edge model is included.
    #   This subroutine is equivalent to what is done in strahl.f between L1043 and L1110
    #   (with a few bug fixes)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ids = 0
    idl = 0
    rx = rvol_lcfs + dbound     # wall (final) location

    # --------------------------------
    # from plad.f
    @inbounds for i in 2:ir
        if rr[i] <= rvol_lcfs
            ids = i + 1 # number of radial points inside of LCFS
        end
        if rr[i] <= (rvol_lcfs + dlim)
            idl = i + 1 # number of radial points inside of limiter
        end
    end

    # ---------------------------------------
    # ions lost at periphery (not parallel) --- NB: ii in main STRAHL code is = ir - 1
    tsu = 0.
    @inbounds for nz in 2:nion
        tsu -= (prox*(diff[ir-1,nz] + diff[ir,nz])*(rn[ir,nz] + ra[ir,nz] -
                rn[ir-1,nz] - ra[ir-1,nz]) + 0.5*(conv[ir-1,nz] + conv[ir,nz])*
                (rn[ir,nz] + ra[ir,nz] + rn[ir-1,nz] + ra[ir-1,nz]))
    end
    tsu = tsu*0.5*pi*rx

    # parallel losses / second

    dsu = 0.
    dsul = 0.

    @inbounds for nz in 2:nion
        for i in ids:idl-1
            dsu += (ra[i,nz] + rn[i,nz])*dv[i]*rr[i]/pro[i]
        end
        for i in idl:ir-1
            dsul += (ra[i,nz] + rn[i,nz])*dv[i]*rr[i]/pro[i]
        end
    end
    dsu = dsu*pi/2. # to divertor
    dsul = dsul*pi/2.   # to limiter

    # time integrated losses at wall/limiters
    tve += (dsul + tsu)*(1.0 - max(0, rcl))*det

    # particles in divertor
    # If recycling is on, particles from limiter and wall come back.
    # Particles in divertor can only return (with rate given by N_div/taudiv) if rcl>=0
    if rcl >= 0 # activated divertor return (rcl>=0) + recycling mode (if rcl>0)
        taustar = 1.0/(1.0/taudiv + 1.0/taupump) # defn, strahl.f L287 # time scale for divertor depletion
        ff = 0.5*det/taustar
        divnew = (divold*(1.0-ff) + (dsu + flx*divbls)*det)/(1.0+ff)    # FS corrected
    else
        divnew = divold + (dsu + flx*divbls)*det
    end

    # particles in pump
    npump += 0.5*(divnew + divold)/taupump*det
    return tve, divnew, npump, tsu, dsu, dsul
end


@inline function linip_arr!(m, n, ts, ys, t, y)
    """
    !
    !     Linear interpolation of time dependent profiles.
    !     For times which are outside the bounds the profile is set to the last/first present time.
    !
    !     Note that this subroutine is different than the one in STRAHL's math.f (L194) because for transport coefficients
    !     we don't care about whether a profile has previously been interpolated or not.
    !
    !     INPUTS:
    !     m                       number of ts-vector elements
    !     n                        number of used rho-vector elements
    !     ts	                The time vector. Values MUST be monotonically increasing.
    !     ys(rho, ts)          array of ordinate values.
    !     t	                The time for which the ordinates shall be interpolated
    !
    !     OUTPUT:
    !     y(rho)                 interpolated profiles at the chosen time
    !

    """

    #if only one time point has been given, then return the same/only value
    #or if the requested time is greater than the maximum time provided, then use the last time
     @inbounds if m == 1 || t >= ts[m]
        for i in 1:n
           y[i] = ys[i,m]
        end
        return
     end

     #if the requested time is smaller than the minimum time provided, then use the first time
     @inbounds if t < ts[1]
        for i in 1:n
           y[i] = ys[i,1]
        end
        return
     end

     #find time point in "ts" vector that is marginally larger than the requested "t"
     @inbounds for k in 1:m+1
        if (t < ts[k+1])
           break
        end
     end

     ff = (t - ts[k])/(ts[k+1] - ts[k])
     @inbounds for i in 1:n
        y[i] = ys[i,k]*(1.0 - ff) + ys[i,k+1]*ff
     end
end
end
