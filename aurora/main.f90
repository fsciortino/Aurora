!MIT License
!
!Copyright (c) 2021 Francesco Sciortino
!Advanced plasma-wall interaction and recycling model provided by Antonello Zito
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
        nion, ir, nt, species, &
        nt_out,  nt_trans, &
        t_trans, D, V, &
        par_loss_rates, &
        src_core, rcl_rad_prof, rfl_rad_prof, spt_rad_prof, en_rec_neut, &
        S_rates, R_rates,  &
        Raxis, rr, pro, qpr, &
        r_saw, dlen,  &
        time, saw, &
        it_out, dsaw, &
        rcl, screen, rcmb, taudiv, taupump, tauwret, &
        S_pump, voldiv, cond, volpump, leak, &
        surfmain, surfdiv, PWI, Zmain, Zdiv, &
        rnmain, rndiv, fluxmain, fluxdiv, ymain, ydiv, depthmain, depthdiv, nmainsat, ndivsat, &
        rvol_lcfs, dbound, dlim, prox, &
        rn_t0, ndiv_t0, npump_t0, nmainwall_t0, ndivwall_t0, alg_opt, evolneut, src_div, &   ! OPTIONAL INPUTS:
        rn_out, &  ! OUT
        N_mainwall, N_divwall, N_div, N_pump, N_out, N_mainret, N_divret, &  ! OUT
        N_tsu, N_dsu, N_dsul, &   !OUT
        rcld_rate, rcld_refl_rate, rcld_recl_rate, rcld_impl_rate, rcld_sput_rate, &   ! OUT
        rclb_rate, rcls_rate, rclp_rate, &   ! OUT
        rclw_rate, rclw_refl_rate, rclw_recl_rate, rclw_impl_rate, rclw_sput_rate)   ! OUT
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
  !     species      integer
  !                    Number of considered background species for the advanced PWI. Not needed for Python calls.
  !     nt_out       integer
  !                    Number of times at which the impurity densities shall be saved
  !     nt_trans     integer
  !                    Number of times at which D,V profiles are given. Not needed for Python calls.
  !     t_trans      real*8 (nt_trans)
  !                    Times at which transport coefficients change [s]
  !     D            real*8 (ir,nt_trans,nion)
  !                    Diffusion coefficient on time and radial grids [cm^2/s]
  !                    This must be given for each charge state and time.
  !     V            real*8 (ir,nt_trans,nion)
  !                    Drift velocity on time and radial grids [cm/s]
  !                    This must be given for each charge state and time
  !     par_loss_rates real*8 (ir,nt)
  !                    Frequency for parallel loss on radial and time grids [1/s]
  !     src_core     real*8 (ir,nt)
  !                    Radial source profile of externally injected neutrals neutrals over time [1/cm^3]
  !     rcl_rad_prof real*8 (ir, nt)
  !                    Normalized radial distribution of impurities re-entering the plasma after recycling
  !                    (or prompt recycling, if the advanced PWI model is used) over time.
  !     rfl_rad_prof real*8 (ir, nt)
  !                    Normalized radial distribution of impurities re-entering the plasma after reflection
  !                    (if the advanced PWI model is used) over time.
  !     spt_rad_prof real*8 (ir, 1+species, nt)
  !                    Normalized radial distribution of impurities re-entering the plasma after sputtering
  !                    (if the advanced PWI model is used) over time.
  !     en_rec_neut  logical
  !                    Logic key for setting energetic reflected/sputtered neutrals 
  !     S_rates      real*8 (ir,nion,nt)
  !                    Ionisation rates (nz=nion must be filled with zeros). Units of [1/s].
  !     R_rates      real*8 (ir,nion,nt)
  !                    Recombination rates (nz=nion must be filled with zeros). Units of [1/s].
  !     Raxis        real*8
  !                    Major radius at the magnetic axis [cm]
  !     rr           real*8 (ir)
  !                    Radial grid, defined using normalized flux surface volumes [cm]
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
  !     screen       real*8
  !                    screening efficiency for the backflow from the divertor reservoir, i.e. fraction
  !                    of rclb_rate which is screened in the SOL/divertor plasma and, therefore, gets
  !                    to be a part of the parallel flow dsu before even re-entering the main plasma
  !     rcmb         real*8
  !                    fraction the impurity ion flow in the SOL which recombines before reaching
  !                    the divertor target, i.e. which enters the divertor neutrals reservoir
  !                    bypassing the divertor wall reservoir.
  !     taudiv       real*8
  !                    Time scale for transport out of the divertor reservoir [s]
  !     taupump      real*8
  !                    Time scale for impurity elimination through out-pumping [s]
  !     tauwret      real*8
  !                    Time scale of temporary retention at the wall [s]
  !     S_pump       real*8
  !                    Pumping speed from the final particle reservoir (divertor or pump chamber) [cm^3/s]
  !     voldiv       real*8
  !                    Volume of the divertor reservoir [cm^3]
  !     cond         real*8
  !                    Conductance between divertor and pump reservoirs [cm^3/s]
  !     volpump      real*8
  !                    Volume of the pumping reservoir [cm^3]
  !     leak         real*8
  !                    Leakage conductance between pump reservoir and main chamber [cm^3/s]
  !     surfmain     real*8
  !                    Effective main wall surface area [cm^2]
  !     surfdiv      real*8
  !                    Effective divertor wall surface area [cm^2] 
  !     PWI          logical
  !                    Logic key for selecting which PWI model to use
  !     Zmain        integer
  !                    Atomic number of the main wall material
  !     Zdiv         integer
  !                    Atomic number of the divertor wall material 
  !     rnmain       real*8 (nt)
  !                    Reflection coefficients for the simulated impurity at the main wall on the time grid
  !     rndiv        real*8 (nt)
  !                    Reflection coefficients for the simulated impurity at the divertor wall on the time grid
  !     fluxmain     real*8 (species,nt)
  !                    Fluxes for each background species onto the main wall on the time grid [s^-1]
  !     fluxdiv      real*8 (species,nt)
  !                    Fluxes for each background species onto the divertor wall on the time grid [s^-1]
  !     ymain        real*8 (1+species,nt)
  !                    Normalized sputtering yields from simulated impurity + background species from the main wall on the time grid
  !     ydiv         real*8 (1+species,nt)
  !                    Normalized sputtering yields from simulated impurity + background species from the divertor wall on the time grid
  !     depthmain    real*8
  !                    Considered impurity implantation depth in the main wall [A]
  !     depthdiv     real*8
  !                    Considered impurity implantation depth in the divertor wall [A]
  !     nmainsat     real*8
  !                    Considered saturation value of the impurity implantation density into the main wall [m^-2]
  !     ndivsat      real*8
  !                    Considered saturation value of the impurity implantation density into the divertor wall [m^-2]  
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
  !     ndiv_t0      real*8, optional
  !                    Impurity density in the divertor reservoir at the start time [1/cm].
  !                    If not provided, this is set to 0.
  !     npump_t0     real*8, optional
  !                    Impurity density in the pump reservoir at the start time [1/cm].
  !                    If not provided, this is set to 0.
  !     nmainwall_t0 real*8, optional
  !                    Impurity density retained at the main wall at the start time [1/cm].
  !                    If not provided, this is set to 0.
  !     ndivwall_t0  real*8, optional
  !                    Impurity density retained at the divertor wall at the start time [1/cm].
  !                    If not provided, this is set to 0.
  !     alg_opt      integer, optional
  !                    Integer to indicate algorithm to be used.
  !                    If set to 0, use the finite-differences algorithm used in the 2018 version of STRAHL.
  !                    If set to 1, use the Linder finite-volume algorithm (see Linder et al. NF 2020)
  !     evolneut     logical, optional  
  !                    Boolean to activate evolution of neutrals (like any ionization stage).
  !                    The D and v given for the 0th charge state apply to these neutrals.
  !     src_div   real*8 (nt), optional
  !                  Flux of particles going into the divertor, given as a function of time in units
  !                  of [1/cm/s]. These particles will only affect the simulation if rcl>=0.
  !                  If not provided, src_div is automatically set to an array of zeros.
  !  
  ! Returns:
  !
  !     rn_out         eal*8 (ir,nion,nt_out)
  !                      Impurity densities (temporarily) in the magnetically-confined plasma at the
  !                      requested times [1/cm^3].
  !     N_mainwall     real*8 (nt_out)
  !                      Impurity densities (permanently) at the main wall over time [1/cm].
  !     N_divwall      real*8 (nt_out)
  !                      Impurity densities (permanently) at the divertor wall over time [1/cm].
  !     N_div          real*8 (nt_out)
  !                      Impurity densities (temporarily) in the divertor reservoir over time [1/cm].
  !     N_pump         real*8 (nt_out)
  !                      Impurity densities (temporarily) in the pump reservoir (if present) over time [1/cm].
  !     N_out          real*8 (nt_out)
  !                      Impurity densities (permanently) removed through the pump [1/cm].
  !     N_mainret      real*8 (nt_out)
  !                      Impurity densities (temporarily) retained at the main wall over time [1/cm].
  !     N_divret       real*8 (nt_out)
  !                      Impurity densities (temporarily) retained at the divertor wall over time [1/cm].
  !     N_tsu          real*8 (nt_out)
  !                      Edge loss [1/cm/s].
  !     N_dsu          real*8 (nt_out)
  !                      Parallel loss [1/cm/s].
  !     N_dsul         real*8 (nt_out)
  !                      Parallel loss to limiter [1/cm/s].
  !     rcld_rate      real*8 (nt_out)
  !                      Total recycling flux from divertor wall [1/cm/s].
  !     rcld_refl_rate real*8 (nt_out)
  !                      Reflected flux from divertor wall [1/cm/s].
  !     rcld_recl_rate real*8 (nt_out)
  !                      Promptly recycled flux from divertor wall [1/cm/s].
  !     rcld_impl_rate real*8 (nt_out)
  !                      Implanted flux into divertor wall [1/cm/s].
  !     rcld_sput_rate real*8 (1+species,nt_out)
  !                      Sputtered fluxes from divertor wall, per each sputtering species [1/cm/s].
  !     rclb_rate      real*8 (nt_out)
  !                      Backflow from divertor neutrals reservoir towards the plasma [1/cm/s].
  !     rcls_rate      real*8 (nt_out)
  !                      Backflow from divertor neutrals reservoir which is screened from the plasma  [1/cm/s].
  !     rclp_rate      real*8 (nt_out)
  !                      Leakage from pump neutrals reservoir (if present) towards the plasma [1/cm/s].
  !     rclw_rate      real*8 (nt_out)
  !                      Total recycling flux from main wall [1/cm/s].
  !     rclw_refl_rate real*8 (nt_out)
  !                      Reflected flux from main wall [1/cm/s].
  !     rclw_recl_rate real*8 (nt_out)
  !                      Promptly recycled flux from main wall [1/cm/s].
  !     rclw_impl_rate real*8 (nt_out)
  !                      Implanted flux into main wall [1/cm/s].
  !     rclw_sput_rate real*8 (1+species,nt_out)
  !                      Sputtered fluxes from main wall, per each sputtering species [1/cm/s].
  ! ---------------------------------------------------------------------------
  
  IMPLICIT NONE

  INTEGER, INTENT(IN)                  :: nion
  INTEGER, INTENT(IN)                  :: ir
  INTEGER, INTENT(IN)                  :: nt
  INTEGER, INTENT(IN)                  :: species
  INTEGER, INTENT(IN)                  :: nt_out   ! required as input
  INTEGER, INTENT(IN)                  :: nt_trans

  REAL*8, INTENT(IN)                   :: t_trans(nt_trans)
  REAL*8, INTENT(IN)                   :: D(ir,nt_trans,nion)
  REAL*8, INTENT(IN)                   :: V(ir,nt_trans,nion)
  REAL*8, INTENT(IN)                   :: par_loss_rates(ir,nt)
  REAL*8, INTENT(IN)                   :: src_core(ir,nt)
  REAL*8, INTENT(IN)                   :: rcl_rad_prof(ir,nt)  
  REAL*8, INTENT(IN)                   :: rfl_rad_prof(ir,nt)
  REAL*8, INTENT(IN)                   :: spt_rad_prof(ir,1+species,nt) 
  LOGICAL, INTENT(IN)                  :: en_rec_neut

  REAL*8, INTENT(IN)                   :: S_rates(ir,nion,nt)
  REAL*8, INTENT(IN)                   :: R_rates(ir,nion,nt)

  REAL*8, INTENT(IN)                   :: Raxis
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

  ! recycling and pumping inputs
  REAL*8, INTENT(IN)                   :: rcl
  REAL*8, INTENT(IN)                   :: screen
  REAL*8, INTENT(IN)                   :: rcmb
  REAL*8, INTENT(IN)                   :: taudiv
  REAL*8, INTENT(IN)                   :: taupump
  REAL*8, INTENT(IN)                   :: tauwret
  REAL*8, INTENT(IN)                   :: S_pump
  REAL*8, INTENT(IN)                   :: voldiv
  REAL*8, INTENT(IN)                   :: cond
  REAL*8, INTENT(IN)                   :: volpump
  REAL*8, INTENT(IN)                   :: leak
  
  ! plasma-wall interaction inputs
  REAL*8, INTENT(IN)                   :: surfmain
  REAL*8, INTENT(IN)                   :: surfdiv
  LOGICAL, INTENT(IN)                  :: PWI
  INTEGER, INTENT(IN)                  :: Zmain
  INTEGER, INTENT(IN)                  :: Zdiv
  REAL*8, INTENT(IN)                   :: rnmain(nt)
  REAL*8, INTENT(IN)                   :: rndiv(nt)
  REAL*8, INTENT(IN)                   :: fluxmain(species,nt)
  REAL*8, INTENT(IN)                   :: fluxdiv(species,nt)
  REAL*8, INTENT(IN)                   :: ymain(1+species,nt)
  REAL*8, INTENT(IN)                   :: ydiv(1+species,nt)
  REAL*8, INTENT(IN)                   :: depthmain
  REAL*8, INTENT(IN)                   :: depthdiv
  REAL*8, INTENT(IN)                   :: nmainsat
  REAL*8, INTENT(IN)                   :: ndivsat

  ! edge
  REAL*8, INTENT(IN)                   :: rvol_lcfs
  REAL*8, INTENT(IN)                   :: dbound
  REAL*8, INTENT(IN)                   :: dlim
  REAL*8, INTENT(IN)                   :: prox

  ! optional arguments
  REAL*8, INTENT(IN), OPTIONAL         :: rn_t0(ir,nion)
  REAL*8, INTENT(IN), OPTIONAL         :: ndiv_t0
  REAL*8, INTENT(IN), OPTIONAL         :: npump_t0
  REAL*8, INTENT(IN), OPTIONAL         :: nmainwall_t0
  REAL*8, INTENT(IN), OPTIONAL         :: ndivwall_t0
  INTEGER, INTENT(IN), OPTIONAL        :: alg_opt
  LOGICAL, INTENT(IN), OPTIONAL        :: evolneut
  REAL*8, INTENT(IN), OPTIONAL         :: src_div(nt)
  
  ! outputs
  REAL*8, INTENT(OUT)                  :: rn_out(ir,nion,nt_out)

  REAL*8, INTENT(OUT)                  :: N_mainwall(nt_out) ! particles stuck permanently at the main wall
  REAL*8, INTENT(OUT)                  :: N_divwall(nt_out)  ! particles stuck permanently at the divertor wall
  REAL*8, INTENT(OUT)                  :: N_div(nt_out)      ! particles in the divertor reservoir
  REAL*8, INTENT(OUT)                  :: N_pump(nt_out)     ! particles in the pump reservoir
  REAL*8, INTENT(OUT)                  :: N_out(nt_out)      ! particles removed through the pump
  REAL*8, INTENT(OUT)                  :: N_mainret(nt_out)  ! particles retained temporarily at the main wall
  REAL*8, INTENT(OUT)                  :: N_divret(nt_out)   ! particles retained temporarily at the divertor wall

  REAL*8, INTENT(OUT)                  :: N_tsu(nt_out)   ! particles lost at the edge
  REAL*8, INTENT(OUT)                  :: N_dsu(nt_out)   ! parallel loss to divertor
  REAL*8, INTENT(OUT)                  :: N_dsul(nt_out)  ! parallel loss to limiter

  REAL*8, INTENT(OUT)                  :: rcld_rate(nt_out)          ! total recycling flux from divertor wall
  REAL*8, INTENT(OUT)                  :: rcld_refl_rate(nt_out)     ! reflected flux from divertor wall
  REAL*8, INTENT(OUT)                  :: rcld_recl_rate(nt_out)     ! promptly recycled flux from divertor wall
  REAL*8, INTENT(OUT)                  :: rcld_impl_rate(nt_out)     ! implanted flux into divertor wall
  REAL*8, INTENT(OUT)                  :: rcld_sput_rate(1+species,nt_out) ! sputtered fluxes from divertor wall
  REAL*8, INTENT(OUT)                  :: rclb_rate(nt_out)          ! non-screened backflow from divertor neutrals reservoir
  REAL*8, INTENT(OUT)                  :: rcls_rate(nt_out)          ! screened backflow from divertor neutrals reservoir
  REAL*8, INTENT(OUT)                  :: rclp_rate(nt_out)          ! leakage from pump neutrals reservoir
  REAL*8, INTENT(OUT)                  :: rclw_rate(nt_out)          ! total recycling flux from main wall
  REAL*8, INTENT(OUT)                  :: rclw_refl_rate(nt_out)     ! reflected flux from main wall
  REAL*8, INTENT(OUT)                  :: rclw_recl_rate(nt_out)     ! promptly recycled flux from main wall
  REAL*8, INTENT(OUT)                  :: rclw_impl_rate(nt_out)     ! implanted flux into main wall
  REAL*8, INTENT(OUT)                  :: rclw_sput_rate(1+species,nt_out) ! sputtered fluxes from main wall
  
  INTEGER     :: i, it, kt, nz, s
  REAL*8      :: rn(ir,nion), ra(ir,nion), dt
  REAL*8      :: Nmainret, Ndivret, Nmainwall, Ndivwall, divnew, pumpnew, nout, divold, pumpold
  REAL*8      :: diff(ir, nion), conv(ir, nion)
  REAL*8      :: tsu, dsu, dsul
  REAL*8      :: rcld, rcld_refl, rcld_recl, div_flux_impl, div_fluxes_sput(1+species)
  REAL*8      :: rclb, rcls, rclp
  REAL*8      :: rclw, rclw_refl, rclw_recl, main_flux_impl, main_fluxes_sput(1+species)
  REAL*8      :: rn_t0_in(ir,nion) ! used to support optional argument rn_t0
  REAL*8      :: ndiv_t0_in ! used to support optional argument ndiv_t0
  REAL*8      :: npump_t0_in ! used to support optional argument npump_t0
  REAL*8      :: nmainwall_t0_in ! used to support optional argument nmainwall_t0
  REAL*8      :: ndivwall_t0_in ! used to support optional argument ndivwall_t0
  REAL*8      :: src_div_in(nt) ! used to support optional argument src_div
  INTEGER     :: sel_alg_opt
  
  ! Only used in impden (define here to avoid re-allocating memory at each impden call)
  REAL*8 :: a(ir,nion), b(ir,nion), c(ir,nion), d1(ir), bet(ir), gam(ir)

  LOGICAL :: evolveneut

  ! rn_t0 is an optional argument. if user does not provide it, set all array elements to 0
  if(present(rn_t0))then
     rn_t0_in=rn_t0
  else
     rn_t0_in=0.0d0 ! all elements set to 0
  endif
  
  ! Initial conditions at the resrvoirs ndiv_t0, npump_t0, nmainwall_t0, ndivwall_t0
  ! are optional arguments. if user does not provide them, set them to 0
  if(present(ndiv_t0))then
     ndiv_t0_in=ndiv_t0
  else
     ndiv_t0_in=0.0d0
  endif
  if(present(npump_t0))then
     npump_t0_in=npump_t0
  else
     npump_t0_in=0.0d0
  endif
  if(present(nmainwall_t0))then
     nmainwall_t0_in=nmainwall_t0
  else
     nmainwall_t0_in=0.0d0
  endif
  if(present(ndivwall_t0))then
     ndivwall_t0_in=ndivwall_t0
  else
     ndivwall_t0_in=0.0d0
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

  ! Set start densities in the plasma
  rn = rn_t0_in  ! all ir, nion points
  
  ! Set start contents in the dynamic reservoirs
  divnew = ndiv_t0_in
  pumpnew = npump_t0_in
  Nmainret = nmainwall_t0_in
  Ndivret = ndivwall_t0_in
  
  ! Initialize permanent reservoirs
  Nmainwall = 0.d0
  Ndivwall = 0.d0
  nout = 0.d0
  
  ! Initialize fluxes
  tsu = 0.0d0
  dsu = 0.0d0
  dsul = 0.0d0

  ! Set starting values in final output arrays
  it = 1
  kt = 1
  if (it_out(it) == 1) then
     !if ( ANY( time_out==time(it) ) ) then

     rn_out(:,:,kt) = rn ! all nion,ir for the first time point
     N_mainwall(kt) = Nmainwall
     N_divwall(kt) = Ndivwall
     N_div(kt) = divnew
     N_pump(kt) = pumpnew
     N_out(kt) = nout
     N_tsu(kt) = tsu
     N_dsu(kt) = dsu
     N_dsul(kt) = dsul

     N_mainret(kt) = Nmainret
     N_divret(kt) = Ndivret
     rcld_rate(kt) = 0.d0
     rcld_refl_rate(kt) = 0.d0
     rcld_recl_rate(kt) = 0.d0
     rcld_impl_rate(kt) = 0.d0
     do s = 1,1+species
        rcld_sput_rate(s,kt) = 0.d0
     end do
     rclb_rate(kt) = 0.d0
     rcls_rate(kt) = 0.d0
     rclp_rate(kt) = 0.d0
     rclw_rate(kt) = 0.d0
     rclw_refl_rate(kt) = 0.d0
     rclw_recl_rate(kt) = 0.d0
     rclw_impl_rate(kt) = 0.d0
     do s = 1,1+species
        rclw_sput_rate(s,kt) = 0.d0
     end do

     kt = kt+1
  end if


  ! ======== time loop: ========
  do it=2,nt
     dt = time(it)-time(it-1)

     ra = rn ! Update old array to new (from previous time step)

     do nz=1,nion
        ! Updated transport coefficients for each charge state
        call linip_arr(nt_trans, ir, t_trans, D(:,:,nz), time(it), diff(:, nz))
        call linip_arr(nt_trans, ir, t_trans, V(:,:,nz), time(it), conv(:,nz))
     end do
     divold = divnew 
     pumpold = pumpnew

     ! Evolve impurity density in the plasma with current transport coefficients
     ! and reservoirs contents according to the imposed recycling parameters
     
     if (sel_alg_opt.eq.0) then
        ! Use old algorithm, just for benchmarking
        call impden0(nion, ir, species, ra, rn,  &   ! IN: old impurity density, OUT: new impurity density
             diff, conv, & 
             par_loss_rates(:,it), &   ! Parallel loss rates for the current time step
             src_core(:,it), &   ! Radial source profile for the current time step
             rcl_rad_prof(:,it), rfl_rad_prof(:,it), spt_rad_prof(:,:,it), en_rec_neut, &   ! Radial recycling profiles for the current time step
             S_rates(:,:,it), R_rates(:,:,it), &   ! Ioniz. and recomb. rates for the current time step
             Raxis, rr, pro, qpr, dlen, &   ! Radial grid parameters
             dt, &   ! full time step
             surfmain, surfdiv, PWI, Zmain, Zdiv, &   ! PWI parameters - surfaces characteristics
             rnmain(it), rndiv(it), &   ! PWI parameters - reflection coeffs. for the current time step
             fluxmain(:,it), fluxdiv(:,it), ymain(:,it), ydiv(:,it), &   ! PWI parameters - background fluxes and sputt. yeilds for the current time step
             depthmain, depthdiv, nmainsat, ndivsat, &   ! PWI parameters - implantation parameters
             rcl, screen, rcmb, &   ! edge parameters
             tsu, dsu, dsul, divold, pumpold, &   ! fluxes and neutrals reservoirs content from previous time step
             taudiv, tauwret, leak, volpump, &   ! pumping parameters
             a, b, c, d1, bet, gam, &   ! re-use memory allocation
             Nmainret, Ndivret, &   ! INOUT: Dynamic wall reservoirs content
             rcld, rcld_refl, rcld_recl, div_flux_impl, div_fluxes_sput(:), &   ! OUT: divertor wall wall recycling fluxes
             rclb, rcls, rclp, &   ! OUT: divertor backflow related fluxes
             rclw, rclw_refl, rclw_recl, main_flux_impl, main_fluxes_sput(:))   ! OUT: main wall recycling fluxes
        
     else
        ! Currently use Linder algorithm for any option other than 0
        call impden1(nion, ir, species, ra, rn,&   ! IN: old impurity density, OUT: new impurity density
             diff, conv, &
             par_loss_rates(:,it), &   ! Parallel loss rates for the current time step
             src_core(:,it), &   ! Radial source profile for the current time step
             rcl_rad_prof(:,it), rfl_rad_prof(:,it), spt_rad_prof(:,:,it), en_rec_neut, &   ! Radial recycling profiles for the current time step
             S_rates(:,:,it), R_rates(:,:,it), &   ! Ioniz. and recomb. rates for the current time step
             Raxis, rr, dlen, &   ! Radial grid parameters
             dt, &   ! renaming dt-->det. In this subroutine, dt is half-step
             surfmain, surfdiv, PWI, Zmain, Zdiv, &   ! PWI parameters - surfaces characteristics
             rnmain(it), rndiv(it), &   ! PWI parameters - reflection coeffs. for the current time step
             fluxmain(:,it), fluxdiv(:,it), ymain(:,it), ydiv(:,it), &   ! PWI parameters - background fluxes and sputt. yeilds for the current time step
             depthmain, depthdiv, nmainsat, ndivsat, &   ! PWI parameters - implantation parameters
             rcl, screen, rcmb, &    ! edge parameters
             tsu, dsu, dsul, divold, pumpold, &    ! fluxes and neutrals reservoirs content from previous time step
             taudiv, tauwret, leak, volpump, &   ! pumping parameters
             evolveneut, &  
             Nmainret, Ndivret, &   ! INOUT: Dynamic wall reservoirs content
             rcld, rcld_refl, rcld_recl, div_flux_impl, div_fluxes_sput(:), &   ! OUT: divertor wall wall recycling fluxes
             rclb, rcls, rclp, &   ! OUT: divertor backflow related fluxes
             rclw, rclw_refl, rclw_recl, main_flux_impl, main_fluxes_sput(:))   ! OUT: main wall recycling fluxes
       
     endif
     
     ! sawteeth
     if (saw(it) == 1) then
        CALL saw_mix(nion, ir, rn, r_saw, dsaw, rr, pro)
     end if

     ! Particle losses towards walls/divertor + Particles permanently absorbed by the walls + Evolution of neutrals reservoirs
     CALL edge_model(nion, ir, ra, rn,  &
          diff, conv, par_loss_rates(:,it), dt, rvol_lcfs, &    ! dt is the full type step here
          dbound, dlim, prox, &
          rr, pro,  &
          PWI, rcl, rcmb, taudiv, taupump, &
          S_pump, voldiv, cond, volpump, leak, &
          src_div_in(it), rcld, rcls, divold, pumpold, &
          divnew, pumpnew, &              ! OUT: update to divold and pumpold
          Nmainwall, Ndivwall, nout, &    ! INOUT: updated values
          tsu, dsu, dsul)                 ! OUT: updated by edge model


     ! Array time-step saving/output
     if (it_out(it) == 1) then
     !if ( ANY( time_out==time(it) ) ) then
        do nz=1,nion
           do i=1,ir
              rn_out(i,nz,kt) = rn(i,nz)
           end do
        end do

        N_mainwall(kt) = Nmainwall
        N_divwall(kt) = Ndivwall
        N_div(kt) = divnew
        N_pump(kt) = pumpnew
        N_out(kt) = nout
        N_tsu(kt) = tsu
        N_dsu(kt) = dsu
        N_dsul(kt) = dsul

        N_mainret(kt) = Nmainret
        N_divret(kt) = Ndivret
        rcld_rate(kt) = rcld
        rcld_refl_rate(kt) = rcld_refl
        rcld_recl_rate(kt) = rcld_recl
        rcld_impl_rate(kt) = div_flux_impl
        do s = 1,1+1
            rcld_sput_rate(s,kt) = div_fluxes_sput(s)
        end do
        rclb_rate(kt) = rclb
        rcls_rate(kt) = rcls
        rclp_rate(kt) = rclp
        rclw_rate(kt) = rclw
        rclw_refl_rate(kt) = rclw_refl
        rclw_recl_rate(kt) = rclw_recl
        rclw_impl_rate(kt) = main_flux_impl
        do s = 1,1+1
            rclw_sput_rate(s,kt) = main_fluxes_sput(s)
        end do

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
    PWI, rcl, rcmb, taudiv, taupump, &
    S_pump, voldiv, cond, volpump, leak, &
    src_div_t, rcld, rcls, divold, pumpold, &
    divnew, pumpnew, Nmainwall, Ndivwall, nout, tsu, dsu, dsul)

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

  LOGICAL, INTENT(IN)                   :: PWI       ! select which PWI model to use
  REAL*8, INTENT(IN)                    :: rcl       ! main/divertor walls recycling coefficient
  REAL*8, INTENT(IN)                    :: rcmb      ! recombination ratio of divertor plasma
  REAL*8, INTENT(IN)                    :: taudiv    ! time scale for divertor retention
  REAL*8, INTENT(IN)                    :: taupump   ! time scale for pumping
  
  REAL*8, INTENT(IN)                    :: S_pump    ! pumping speed
  REAL*8, INTENT(IN)                    :: voldiv    ! divertor volume
  REAL*8, INTENT(IN)                    :: cond      ! divertor-pump conductance
  REAL*8, INTENT(IN)                    :: volpump   ! pump volume
  REAL*8, INTENT(IN)                    :: leak      ! leaking conductance from pump towards main chamber

  REAL*8, INTENT(IN)                    :: src_div_t ! injected flux into the div. reservoir
  REAL*8, INTENT(IN)                    :: rcld      ! recycling flux from the div. wall into the div. reservoir
  REAL*8, INTENT(IN)                    :: rcls      ! portion of backflow from divertor which is screened
  REAL*8, INTENT(IN)                    :: divold    ! particles initially in the div. reservoir (to update)
  REAL*8, INTENT(IN)                    :: pumpold   ! particles initially in the pump reservoir (to update)

  REAL*8, INTENT(OUT)                   :: divnew    ! particles in the div. reservoir (updated)
  REAL*8, INTENT(OUT)                   :: pumpnew   ! particles in the pump reservoir (updated)
  REAL*8, INTENT(INOUT)                 :: Nmainwall ! particles stuck at main wall (updated)
  REAL*8, INTENT(INOUT)                 :: Ndivwall  ! particles stuck at divertor wall (updated)
  REAL*8, INTENT(INOUT)                 :: nout      ! particles removed through the pump (updated)
  REAL*8, INTENT(OUT)                   :: tsu       ! edge loss
  REAL*8, INTENT(OUT)                   :: dsu       ! parallel loss
  REAL*8, INTENT(OUT)                   :: dsul      ! parallel loss to limiter

  INTEGER :: i, nz, ids, idl, ids1, idl1
  REAL*8 :: rx, pi, taustar_div, taustar_pump, taucond_eff_back, taucond_eff_forw, tauleak, taupump_eff, ff_div, ff_pump

  ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Compute edge fluxes given multi-reservoir parameters
  ! Core-densities do not directly depend on this -- but recycling
  ! can only be activated if this 1D edge model is included.
  ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  pi = 4. * atan(1.)
  rx=rvol_lcfs+dbound   ! wall (final) location

  ! ---------------------------------------------

  do i=2,ir
     if(rr(i) .le. rvol_lcfs) ids=i+1   ! number of radial points inside of LCFS
     if(rr(i) .le. (rvol_lcfs+dlim)) idl=i+1     ! number of radial points inside of limiter
  end do
  ids1=ids-1
  idl1 = idl-1 

  ! --------------------------------------
  
  ! Ions lost at the plasma periphery (radial edge loss) / seconds
  tsu=0.d0
  do nz=2,nion
     tsu=tsu - prox * (diff(ir-1,nz)+diff(ir,nz)) * (rn(ir,nz)+ra(ir,nz) - rn(ir-1,nz)-ra(ir-1,nz))  + &
          .5*(conv(ir-1,nz)+conv(ir,nz)) *(  rn(ir,nz)+ra(ir,nz)+ rn(ir-1,nz)+ra(ir-1,nz) )
  end do
  tsu=tsu*.5*pi*rx

  !  Parallel ion loss in the SOL (towards divertor) / seconds
  dsu=0.d0
  do nz=2,nion
     do i=ids,idl1
        dsu=dsu+(ra(i,nz)+rn(i,nz)) *par_loss_rate(i)*rr(i)/pro(i)
     end do
  end do
  dsu = dsu*pi/2.

  !  Parallel ion loss in the SOL (towards limiter) / seconds
  dsul=0.d0
  do nz=2,nion
     do i=idl,ir-1
        dsul=dsul+(ra(i,nz)+rn(i,nz)) *par_loss_rate(i)*rr(i)/pro(i)
     end do
  end do
  dsul = dsul*pi/2.

  ! Permanent particle losses at the walls
  
  ! Advanced plasma-wall interaction model activated: permanent wall reservoir not used
  if (PWI) then
  
     Nmainwall = Nmainwall  ! permanent main wall reservoir always empty
     Ndivwall = Ndivwall  ! permanent divertor wall reservoir always empty
  
  ! Simple plasma-wall interaction model activated: permanent wall reservoir used
  else
  
     if (rcl.ge.0) then
        Nmainwall = Nmainwall + (dsul + tsu) * (1.-rcl)*det      ! rcl=0 or rcl>0, but always w/ divertor return
        Ndivwall = Ndivwall + (dsu+rcls) * (1.-rcmb) * (1.-rcl)*det   ! rcl=0 or rcl>0, but always w/ divertor return
     else
        Nmainwall = Nmainwall + (dsul + tsu) * det      ! no recycling, no divertor return
        Ndivwall = Ndivwall + (dsu+rcls) * (1.-rcmb) * det   ! no recycling, no divertor return
     endif
  
  endif

  ! Evolution of particle content in divertor/pump reservoirs
  ! If recycling is on (rcl>0), a fraction of particles hitting the walls can come back
  ! Particles in divertor can only return to the plasma (with rate given by N_div/taudiv) if rcl>=0
  
  ! Activated divertor backflow (rcl>=0) + recycling mode (if rcl>0)
  if (rcl.ge.0) then
     
     ! Adimensional model (pumping done from the divertor and defined by a time taupump)
     if (voldiv.eq.0) then
        taustar_div = 1./(1./taudiv+1./taupump)  ! eff. time scale for divertor depletion
        ff_div = .5*det/taustar_div
        divnew = ( divold*(1.-ff_div) + ((dsu+rcls)*rcmb + rcld + src_div_t)*det )/(1.+ff_div)
        pumpnew = pumpold
    
     ! Physical volumes for the reservoirs
     else
        
        ! Pumping done from the divertor and defined by a speed S_pump
        if (volpump.eq.0) then
           taupump_eff = voldiv/S_pump
           taustar_div = 1./(1./taudiv+1./taupump_eff)  ! eff. time scale for divertor depletion
           ff_div = .5*det/taustar_div
           divnew = ( divold*(1.-ff_div) + ((dsu+rcls)*rcmb + rcld + src_div_t)*det )/(1.+ff_div)
           pumpnew = pumpold
        
        ! Pumping done from a secondary pump chamber via a conductance cond and defined by a speed S_pump
        else
           taupump_eff = volpump/S_pump
           taucond_eff_forw = voldiv/cond
           taucond_eff_back = volpump/cond
           taustar_div = 1./(1./taudiv+1./taucond_eff_forw)  ! eff. time scale for flow divertor --> pump
           ff_div = .5*det/taustar_div
           divnew = ( divold*(1.-ff_div) + ((dsu+rcls)*rcmb + rcld + src_div_t + (pumpold/taucond_eff_back))*det )/(1.+ff_div)
           if (leak.eq.0) then
              taustar_pump = 1./(1./taucond_eff_back+1./taupump_eff)  ! eff. time scale for pump depletion
           else
              tauleak = volpump/leak
              taustar_pump = 1./(1./taucond_eff_back+1./taupump_eff+1./tauleak)  ! eff. time scale for pump depletion
           endif
           ff_pump = .5*det/taustar_pump
           pumpnew = ( pumpold*(1.-ff_pump) + (divold/taucond_eff_forw)*det )/(1.+ff_pump)
        endif
     endif
  
  ! No backflow from the divertor (rcl<0), i.e. taudiv --> inf
  else
     
     ! Adimensional model (pumping done from the divertor and defined by a time taupump)
     if (voldiv.eq.0) then
        taustar_div = taupump  ! eff. time scale for divertor depletion
        ff_div = .5*det/taustar_div
        divnew = ( divold*(1.-ff_div) + ((dsu+rcls)*rcmb + rcld + src_div_t)*det )/(1.+ff_div)  
        pumpnew = pumpold
     
     ! Physical volumes for the reservoirs
     else
        
        ! Pumping done from the divertor and defined by a speed S_pump
        if (volpump.eq.0) then
           taupump_eff = voldiv/S_pump
           taustar_div = taupump_eff  ! eff. time scale for divertor depletion
           ff_div = .5*det/taustar_div
           divnew = ( divold*(1.-ff_div) + ((dsu+rcls)*rcmb + rcld + src_div_t)*det )/(1.+ff_div)
           pumpnew = pumpold
        
        ! Pumping done from a secondary pump chamber via a conductance cond and defined by a speed S_pump
        else
           taupump_eff = volpump/S_pump
           taucond_eff_forw = voldiv/cond
           taucond_eff_back = volpump/cond
           taustar_div = taucond_eff_forw  ! eff. time scale for flow divertor --> pump
           ff_div = .5*det/taustar_div
           divnew = ( divold*(1.-ff_div) + ((dsu+rcls)*rcmb + rcld + src_div_t + (pumpold/taucond_eff_back))*det )/(1.+ff_div)
           if (leak.eq.0) then
              taustar_pump = 1./(1./taucond_eff_back+1./taupump_eff)  ! eff. time scale for pump depletion
           else
              tauleak = volpump/leak
              taustar_pump = 1./(1./taucond_eff_back+1./taupump_eff+1./tauleak)  ! eff. time scale for pump depletion
           endif
           ff_pump = .5*det/taustar_pump
           pumpnew = ( pumpold*(1.-ff_pump) + (divold/taucond_eff_forw)*det )/(1.+ff_pump)
        
        endif
     
     endif
     
  endif
  
  ! Evolution of particles permanently pumped

  ! Adimensional model (pumping done from the divertor and defined by a time taupump)
  if (voldiv.eq.0) then
     nout = nout + .5*(divnew+divold)/taupump*det
  
  ! Physical volumes for the reservoirs
  else
  
     ! Pumping done from the divertor and defined by a speed S_pump
     if (volpump.eq.0) then
        nout = nout + .5*(divnew+divold)/taupump_eff*det
     
     ! Pumping done from a secondary pump chamber via a conductance cond and defined by a speed S_pump
     else
        nout = nout + .5*(pumpnew+pumpold)/taupump_eff*det   
     
     endif
  
  endif

  return

end subroutine edge_model
