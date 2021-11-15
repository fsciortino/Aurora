!MIT License
!
!Copyright (c) 2021 Francesco Sciortino
!
!Permission is hereby granted, free of charge, to any person obtaining a copy
!of this software and associated documentation files (the "Software"), to deal
!in the Software without restriction, including without limitation the rights
!to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
!copies of the Software, and to permit persons to whom the Software is
!furnished to do so, subject to the following conditions:
!
!The above copyright notice and this permission notice shall be included in all
!copies or substantial portions of the Software.
!
!THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
!IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
!FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
!AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
!LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
!OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
!SOFTWARE.


subroutine run(  &
        nion, ir, nt, &
        nt_out,  nt_trans, &
        t_trans, D, V, &
        par_loss_rates, &
        src_core, rcl_rad_prof, &
        S_rates, R_rates,  &
        rr, pro, qpr, &
        r_saw, dlen,  &
        time, saw, &
        it_out, dsaw, &
        rcl, taudiv, taupump, tauwret, &
        rvol_lcfs, dbound, dlim, prox, &
        rn_t0, alg_opt, evolneut, src_div, &   ! OPTIONAL INPUTS:
        rn_out, &  ! OUT
        N_wall, N_div, N_pump, N_ret, &  ! OUT
        N_tsu, N_dsu, N_dsul,&   !OUT
        rcld_rate, rclw_rate)   ! OUT
  !
  ! Run forward model of radial impurity transport, returning the density of
  ! each charge state over time and space. 
  !
  ! ------ All inputs are in CGS units -----
  !
  ! Get list of required input parameters in Python using 
  ! print(aurora.run.__doc__)
  !
  ! Args:
  !     nion         integer
  !                    Number of ionization stages. Not needed for Python calls.
  !     ir           integer
  !                    Number of radial grid points. Not needed for Python calls.
  !     nt           integer
  !                    Number of time steps for the solution. Not needed for Python calls.
  !     nt_out       integer
  !                    Number of times at which the impurity densities shall be saved.
  !     nt_trans     integer
  !                    Number of times at which D,V profiles are given. Not needed for Python calls.
  !     t_trans      real*8 (nt_trans)
  !                    Times at which transport coefficients change [s].
  !     D            real*8 (ir,nt_trans,nion)
  !                    Diffusion coefficient on time and radial grids [cm^2/s]
  !                    This must be given for each charge state and time.
  !     V            real*8 (ir,nt_trans,nion)
  !                    Drift velocity on time and radial grids [cm/s]
  !                    This must be given for each charge state and time.
  !     par_loss_rates  real*8 (ir,nt)
  !                    Frequency for parallel loss on radial and time grids [1/s]
  !     src_core real*8 (ir,nt)
  !                    Radial profile of neutrals over time.
  !     rcl_rad_prof real*8 (ir, nt)
  !                    Radial distribution of impurities re-entering the core reservoir after recycling,
  !                    given as a function of time.
  !                    NB: this should be a normalized profile!
  !     S_rates      real*8 (ir,nion,nt)
  !                    Ionisation rates (nz=nion must be filled with zeros).
  !     R_rates      real*8 (ir,nion,nt)
  !                    Recombination rates (nz=nion must be filled with zeros)
  !     rr           real*8 (ir)
  !                    Radial grid, defined using normalized flux surface volumes
  !     pro          real*8 (ir)
  !                    Normalized first derivative of the radial grid, defined by
  !                    pro = (drho/dr)/(2 d_rho) = rho'/(2 d_rho)
  !     qpr          real*8 (ir)
  !                    Normalized second derivative of the radial grid, defined as
  !                    qpr = (d^2 rho/dr^2)/(2 d_rho) = rho''/(2 d_rho)
  !     r_saw        real*8
  !                    Sawteeth inversion radius [cm]
  !     dlen         real*8
  !                    Decay length at last radial grid point
  !     time         real*8 (nt)
  !                    Time grid for transport solver
  !     saw          integer (nt)
  !                    Switch to induce a sawtooth crashes
  !                    If saw(it) eq 1 there is a crash at time(it)
  !     it_out       integer (nt)
  !                    Store the impurity distributions if it_out(it).eq.1
  !     dsaw         real*8
  !                    Width of sawtooth crash region.
  !     rcl          real*8
  !                    Wall recycling coefficient. Normally, this would be in the range [0,1].
  !                    However, if set to a value <0, then this is interpreted as a flag, indicating
  !                    that particles in the divertor should NEVER return to the main plasma.
  !                    This is effectively what the rclswitch flag does in STRAHL (confusingly).
  !     taudiv       real*8
  !                    Time scale for transport out of the divertor reservoir [s]
  !     taupump      real*8
  !                    Time scale for impurity elimination through out-pumping [s]
  !     tauwret      real*8
  !                    Time scale of temporary retention at the wall [s]
  !     rvol_lcfs    real*8
  !                    Radius (in rvol units, cm) at which the LCFS is located
  !     dbound       real*8
  !                    Width of the SOL, given by r_bound - r_lcfs (in rvol coordinates, cm)
  !                    This value sets the width of the radial grid.
  !     dlim         real*8
  !                    Position of the limiter wrt to the LCFS, i.e. r_lim - r_lcfs (cm, in r_vol units).
  !                    Inside of this limiter location, the parallel connection length to the divertor applies,
  !                    while outside of it the relevant connection length is the one to the limiter.
  !                    These different connection lengths must be taken into consideration when
  !                    preparing the parallel loss rate variable.
  !     prox         real*8
  !                    Grid parameter for loss rate at the last radial point, returned by
  !                    `get_radial_grid' subroutine.
  !
  !
  !     OPTIONAL ARGUMENTS
  !
  !     rn_t0        real*8 (ir,nion), optional
  !                    Impurity densities at the start time [1/cm^3].
  !                    If not provided, all elements are set to 0.
  !     alg_opt      integer, optional
  !                    Integer to indicate algorithm to be used.
  !                    If set to 0, use the finite-differences algorithm used in the 2018 version of STRAHL.
  !                    If set to 1, use the Linder finite-volume algorithm (see Linder et al. NF 2020)
  !     evolneut     logical, optional  
  !                    Boolean to activate evolution of neutrals (like any ionization stage).
  !                    The D and v given for the 0th charge state apply to these neutrals.
  !     src_div   real*8 (nt), optional
  !                  Flux of particles going into the divertor, given as a function of time.
  !                  These particles will only affect the simulation if rcl>=0.
  !                  If not provided, src_div is automatically set to an array of zeros.
  !  
  ! Returns:
  !
  !     rn_out       real*8 (ir,nion,nt_out)
  !                    Impurity densities (temporarily) in the magnetically-confined plasma at the
  !                    requested times [1/cm^3].
  !     N_ret        real*8 (nt_out)
  !                    Impurity densities (permanently) retained at the wall over time [1/cm^3].
  !     N_wall       real*8 (nt_out)
  !                    Impurity densities (temporarily) at the wall over time [1/cm^3].
  !     N_div        real*8 (nt_out)
  !                    Impurity densities (temporarily) in the divertor reservoir over time [1/cm^3].
  !     N_pump       real*8 (nt_out)
  !                    Impurity densities (permanently) in the pump [1/cm^3].
  !     N_tsu        real*8 (nt_out)
  !                    Edge loss [1/cm^3].
  !     N_dsu        real*8 (nt_out)
  !                    Parallel loss [1/cm^3].
  !     N_dsul       real*8 (nt_out)
  !                    Parallel loss to limiter [1/cm^3].
  !     rcld_rate    real*8 (nt_out)
  !                    Recycling from divertor [1/cm^3/s].
  !     rclw_rate    real*8 (nt_out)
  !                    Recycling from wall [1/cm^3/s].
  ! ---------------------------------------------------------------------------

  IMPLICIT NONE

  INTEGER, INTENT(IN)                  :: nion
  INTEGER, INTENT(IN)                  :: ir
  INTEGER, INTENT(IN)                  :: nt
  INTEGER, INTENT(IN)                  :: nt_out   ! required as input
  INTEGER, INTENT(IN)                  :: nt_trans

  REAL*8, INTENT(IN)                   :: t_trans(nt_trans)
  REAL*8, INTENT(IN)                   :: D(ir,nt_trans,nion)
  REAL*8, INTENT(IN)                   :: V(ir,nt_trans,nion)
  REAL*8, INTENT(IN)                   :: par_loss_rates(ir,nt)
  REAL*8, INTENT(IN)                   :: src_core(ir,nt)
  REAL*8, INTENT(IN)                   :: rcl_rad_prof(ir,nt)  

  REAL*8, INTENT(IN)                   :: S_rates(ir,nion,nt)
  REAL*8, INTENT(IN)                   :: R_rates(ir,nion,nt)

  REAL*8, INTENT(IN)                   :: rr(ir)
  REAL*8, INTENT(IN)                   :: pro(ir)
  REAL*8, INTENT(IN)                   :: qpr(ir)

  REAL*8, INTENT(IN)                   :: r_saw
  REAL*8, INTENT(IN)                   :: dlen

  REAL*8, INTENT(IN)                   :: time(nt)
!!!  REAL*8, INTENT(IN)                   :: time_out(nt_out)

  INTEGER, INTENT(IN)                  :: saw(nt)
  INTEGER, INTENT(IN)                  :: it_out(nt)   !!!

  REAL*8, INTENT(IN)                   :: dsaw

  ! recycling inputs
  REAL*8, INTENT(IN)                   :: rcl
  REAL*8, INTENT(IN)                   :: taudiv
  REAL*8, INTENT(IN)                   :: taupump
  REAL*8, INTENT(IN)                   :: tauwret

  ! edge
  REAL*8, INTENT(IN)                   :: rvol_lcfs
  REAL*8, INTENT(IN)                   :: dbound
  REAL*8, INTENT(IN)                   :: dlim
  REAL*8, INTENT(IN)                   :: prox

  ! OPTIONAL ARGUMENTS
  REAL*8, INTENT(IN), OPTIONAL         :: rn_t0(ir,nion)
  INTEGER, INTENT(IN), OPTIONAL        :: alg_opt
  LOGICAL, INTENT(IN), OPTIONAL        :: evolneut
  REAL*8, INTENT(IN), OPTIONAL         :: src_div(nt)
  
  ! outputs
  REAL*8, INTENT(OUT)                  :: rn_out(ir,nion,nt_out)

  REAL*8, INTENT(OUT)                  :: N_wall(nt_out)   ! particles at wall
  REAL*8, INTENT(OUT)                  :: N_div(nt_out)   ! particles in divertor
  REAL*8, INTENT(OUT)                  :: N_pump(nt_out) ! particles in pump
  REAL*8, INTENT(OUT)                  :: N_ret(nt_out)   ! particles retained indefinitely at wall

  REAL*8, INTENT(OUT)                  :: N_tsu(nt_out)   ! particles lost at the edge
  REAL*8, INTENT(OUT)                  :: N_dsu(nt_out)   ! parallel loss to divertor
  REAL*8, INTENT(OUT)                  :: N_dsul(nt_out)   ! parallel loss to limiter

  REAL*8, INTENT(OUT)                  :: rcld_rate(nt_out)   ! recycling from divertor
  REAL*8, INTENT(OUT)                  :: rclw_rate(nt_out)   ! recycling from wall
  
  INTEGER     :: i, it, kt, nz
  REAL*8      :: rn(ir,nion), ra(ir,nion), dt
  REAL*8      :: Nret, tve, divnew, npump, divold
  REAL*8      :: diff(ir, nion), conv(ir, nion)
  REAL*8      :: tsu, dsu, dsul
  REAL*8      :: rcld, rclw
  REAL*8      :: rn_t0_in(ir,nion) ! used to support optional argument rn_t0
  REAL*8      :: src_div_in(nt) ! used to support optional argument src_div
  INTEGER     :: sel_alg_opt
  
  ! Only used in impden (define here to avoid re-allocating memory at each impden call)
  REAL*8 :: a(ir,nion), b(ir,nion), c(ir,nion), d1(ir), bet(ir), gam(ir)

  LOGICAL :: evolveneut
    
  ! rn_time0 is an optional argument. if user does not provide it, set all array elements to 0
  if(present(rn_t0))then
     rn_t0_in=rn_t0
  else
     rn_t0_in=0.0d0 ! all elements set to 0
  endif

  if(present(alg_opt))then
     sel_alg_opt=alg_opt
  else
     sel_alg_opt=1 ! use Linder algorithm by default
  endif

  if(present(evolneut))then
     evolveneut=evolneut
  else
     evolveneut=.false.
  endif

  if (present(src_div))then
     src_div_in = src_div
  else
     src_div_in = 0.0d0 ! all elements set to 0
  endif
  
  ! initialize edge quantities
  Nret=0.d0
  tve = 0.d0
  divnew = 0.0d0
  npump = 0.d0
  tsu = 0.0d0
  dsu = 0.0d0
  dsul = 0.0d0

  ! set start densities
  rn = rn_t0_in  ! all ir, nion points

  ! Set starting values in final output arrays
  it = 1
  kt = 1
  if (it_out(it) == 1) then
     !if ( ANY( time_out==time(it) ) ) then

     rn_out(:,:,kt) = rn ! all nion,ir for the first time point
     N_wall(kt) = tve
     N_div(kt) = divnew
     N_pump(kt) = npump
     N_tsu(kt) = tsu
     N_dsu(kt) = dsu
     N_dsul(kt) = dsul

     N_ret(kt) = Nret
     rcld_rate(kt) = 0.d0
     rclw_rate(kt) = 0.d0

     kt = kt+1
  end if


  ! ======== time loop: ========
  do it=2,nt
     dt = time(it)-time(it-1)

     ra = rn ! update old array to new (from previous time step)

     do nz=1,nion
        ! updated transport coefficients for each charge state
        call linip_arr(nt_trans, ir, t_trans, D(:,:,nz), time(it), diff(:, nz))
        call linip_arr(nt_trans, ir, t_trans, V(:,:,nz), time(it), conv(:,nz))
     end do
     divold = divnew 

     ! evolve impurity density with current transport coeffs
     if (sel_alg_opt.eq.0) then
        ! Use old algorithm, just for benchmarking
        call impden0( nion, ir, ra, rn,  &   !OUT: rn
             diff, conv, par_loss_rates(:,it), src_core(:,it), &
             rcl_rad_prof(:,it), &
             S_rates(:,:,it), R_rates(:,:,it),  &
             rr, pro, qpr, &
             dlen,  &
             dt, &   ! full time step
             rcl, tsu, dsul, divold, & ! tsu,dsul,divnew from previous recycling step
             taudiv, tauwret, &
             a, b, c, d1, bet, gam, &  ! re-use memory allocation
             Nret, &         ! INOUT: Nret
             rcld, rclw )    ! OUT: rcld, rclw
        
     else   ! currently use Linder algorithm for any option other than 0
        ! Linder algorithm
        call impden1(nion, ir, ra, rn,&
             diff, conv, par_loss_rates(:,it), &
             src_core(:,it), rcl_rad_prof(:,it), &
             S_rates(:,:,it), R_rates(:,:,it),  &
             rr, &
             dlen, &
             dt,  &    ! renaming dt-->det. In this subroutine, dt is half-step
             rcl, tsu, dsul, divold, &
             taudiv, tauwret, &
             evolveneut, &  
             Nret, rcld, rclw)
       
     endif
     
     ! sawteeth
     if (saw(it) == 1) then
        CALL saw_mix(nion, ir, rn, r_saw, dsaw, rr, pro)
     end if

     ! particle losses at wall & divertor + recycling
     CALL edge_model(nion, ir, ra, rn,  &
          diff, conv, par_loss_rates(:,it), dt, rvol_lcfs, &    ! dt is the full type step here
          dbound, dlim, prox, &
          rr, pro,  &
          rcl, taudiv, taupump, &
          src_div_in(it), divold, &
          divnew, &        ! OUT: update to divold
          tve, npump, &     ! INOUT: updated values
          tsu, dsu, dsul)  ! OUT: updated by edge model


     ! array time-step saving/output
     if (it_out(it) == 1) then
     !if ( ANY( time_out==time(it) ) ) then
        do nz=1,nion
           do i=1,ir
              rn_out(i,nz,kt) = rn(i,nz)
           end do
        end do

        N_wall(kt) = tve
        N_div(kt) = divnew
        N_pump(kt) = npump
        N_tsu(kt) = tsu
        N_dsu(kt) = dsu
        N_dsul(kt) = dsul

        N_ret(kt) = Nret
        rcld_rate(kt) = rcld
        rclw_rate(kt) = rclw

        kt = kt+1
     end if

  end do
  ! ====== end of time loop ========

  return
end subroutine run







subroutine saw_mix(nion, ir, rn, rsaw, dsaw, rr, pro)

  IMPLICIT NONE

  INTEGER, INTENT(IN)                      :: nion
  INTEGER, INTENT(IN)                      :: ir
  REAL*8, INTENT(INOUT)                 :: rn(ir,nion)
  REAL*8, INTENT(IN)                       :: rsaw
  REAL*8, INTENT(IN)                       :: dsaw
  REAL*8, INTENT(IN)                       :: rr(ir)
  REAL*8, INTENT(IN)                       :: pro(ir)

  INTEGER :: i, nz, imix

  REAL*8 sum , sum_old, ff

  !     index of mixing radius
  imix=0
  do i=1,ir
     if (rr(i) > rsaw .and. imix == 0) then
        imix = i
     end if
  end do

  do nz=2,nion              !loop over ionized stages

     !     area integral in mixing radius of old profile

     sum_old =0.125*(rn(imix,nz)*rr(imix)/pro(imix)  &  ! only use new density, rn
          - rn(imix-1,nz)*rr(imix-1)/pro(imix-1))
     do i=2,imix-1
        sum_old = sum_old + rn(i,nz)*rr(i)/pro(i)
     end do

     !    ERFC sawtooth crash model
     ff = sum_old/rr(imix)**2  ! nmean
     do i=1, ir
        rn(i,nz) = ff/2. * erfc(( rr(i) - rsaw )/dsaw)+(rn(i,nz)/2.0 )*erfc(-(rr(i)-rsaw)/dsaw)
     end do

     !      flat profile
     !  ff = sum_old/rr(imix)**2
     !  do i=1,imix-1
     !    rn(i,nz) = ff
     !  end do
     !  rn(imix,nz) = (ra(imix+1,nz)+ff)/2.

     !      area integral in mixing radius of new profile

     sum =0.125*( rn(imix, nz)*rr(imix) /pro(imix) -  &
          rn(imix-1, nz)*rr(imix-1)/pro(imix-1))
     do i=2,imix-1
        sum = sum + rn(i,nz)*rr(i)/pro(i)
     end do

     !      ensure particle conservation

     ff = sum_old/sum
     do i=1,imix
        rn(i,nz) = rn(i,nz)*ff
     end do

  end do

  return
end subroutine saw_mix







subroutine edge_model( &
    nion, ir, ra, rn, &
    diff, conv, &
    par_loss_rate, det, rvol_lcfs, &
    dbound, dlim, prox, &
    rr, pro,  &
    rcl, taudiv, taupump, &
    src_div_t, divold, &
    divnew, tve, npump, tsu, dsu, dsul)

  IMPLICIT NONE

  INTEGER, INTENT(IN)                   :: nion
  INTEGER, INTENT(IN)                   :: ir
  REAL*8, INTENT(INOUT)                 :: ra(ir,nion)
  REAL*8, INTENT(INOUT)                 :: rn(ir,nion)

  REAL*8, INTENT(IN)                    :: diff(ir, nion)
  REAL*8, INTENT(IN)                    :: conv(ir, nion)

  REAL*8, INTENT(IN)                    :: par_loss_rate(ir)
  REAL*8, INTENT(IN)                    :: det   ! full time step
  REAL*8, INTENT(IN)                    :: rvol_lcfs 
  REAL*8, INTENT(IN)                    :: dbound
  REAL*8, INTENT(IN)                    :: dlim
  REAL*8, INTENT(IN)                    :: prox  ! for edge loss calculation

  REAL*8, INTENT(IN)                    :: rr(ir)
  REAL*8, INTENT(IN)                    :: pro(ir)

  REAL*8, INTENT(IN)                    :: rcl
  REAL*8, INTENT(IN)                    :: taudiv  !time scale for divertor
  REAL*8, INTENT(IN)                    :: taupump !time scale for pump

  REAL*8, INTENT(IN)                    :: src_div_t ! injected flux into the divertor 
  REAL*8, INTENT(IN)                    :: divold !particles initially in divertor (to update)

  REAL*8, INTENT(OUT)                   :: divnew !particles in divertor (updated)
  REAL*8, INTENT(INOUT)                 :: tve   !particles at wall (updated)
  REAL*8, INTENT(INOUT)                 :: npump !particles in pump (updated)
  REAL*8, INTENT(OUT)                   :: tsu   !edge loss
  REAL*8, INTENT(OUT)                   :: dsu   !parallel loss
  REAL*8, INTENT(OUT)                   :: dsul   !parallel loss to limiter

  INTEGER :: i, nz, ids, idl, ids1, idl1
  REAL*8 :: rx, pi, taustar, ff

  ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Compute edge fluxes given multi-reservoir parameters
  ! Core-densities do not directly depend on this -- but recycling
  ! can only be activated if this 1D edge model is included.
  ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  pi = 4. * atan(1.)
  rx=rvol_lcfs+dbound   ! wall (final) location

  ! ---------------------------------------------
  ! from plad.f
  do i=2,ir
     if(rr(i) .le. rvol_lcfs) ids=i+1   ! number of radial points inside of LCFS
     if(rr(i) .le. (rvol_lcfs+dlim)) idl=i+1     ! number of radial points inside of limiter
  enddo
  ids1=ids-1
  idl1 = idl-1 

  ! --------------------------------------
  !  ions lost at periphery (not parallel)
  tsu=0.d0
  do nz=2,nion
     tsu=tsu - prox * (diff(ir-1,nz)+diff(ir,nz)) * (rn(ir,nz)+ra(ir,nz) - rn(ir-1,nz)-ra(ir-1,nz))  + &
          .5*(conv(ir-1,nz)+conv(ir,nz)) *(  rn(ir,nz)+ra(ir,nz)+ rn(ir-1,nz)+ra(ir-1,nz) )
  end do
  tsu=tsu*.5*pi*rx

  !  parallel losses / second
  dsu=0.d0
  do nz=2,nion
     do i=ids,idl1
        dsu=dsu+(ra(i,nz)+rn(i,nz)) *par_loss_rate(i)*rr(i)/pro(i)
     end do
  end do
  dsu = dsu*pi/2.  ! to divertor

  dsul=0.d0
  do nz=2,nion
     do i=idl,ir-1
        dsul=dsul+(ra(i,nz)+rn(i,nz)) *par_loss_rate(i)*rr(i)/pro(i)
     end do
  end do
  dsul = dsul*pi/2.  ! to limiter

  ! time integrated losses at wall/limiters
  if (rcl.ge.0) then
     tve = tve + (dsul + tsu) * (1.-rcl)*det   ! rcl=0 or rcl>0, but always w/ divertor return
  else
     tve = tve + (dsul + tsu) * det  ! no recycling, no divertor return
  endif

  ! particles in divertor
  ! If recycling is on, particles from limiter and wall come back.
  ! Particles in divertor can only return (with rate given by N_div/taudiv) if rcl>=0
  if (rcl.ge.0) then  ! activated divertor return (rcl>=0) + recycling mode (if rcl>0)
     taustar = 1./(1./taudiv+1./taupump)    ! time scale for divertor depletion
     ff = .5*det/taustar
     
     divnew = ( divold*(1.-ff) + (dsu + src_div_t)*det )/(1.+ff)
  else
     divnew = divold + (dsu+src_div_t)*det
  endif

  ! particles in pump
  npump = npump + .5*(divnew+divold)/taupump*det

  return
end subroutine edge_model
