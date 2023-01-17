!MIT License
!
!Copyright (c) 2021 Francesco Sciortino
!
!Extended recycling model and advanced plasma-wall interaction model
!provided by Antonello Zito
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

! Subroutines to forward model impurity transport. 
!
! impden0 uses the standard finite-difference scheme used in STRAHL.
! impden1 uses the finite-volumes scheme described in Linder NF 2020. 
!

subroutine impden0(nion, ir, species, ra, rn, diff, conv, par_loss_rate, &
     src_prof, rcl_rad_prof, rfl_rad_prof, spt_rad_prof, en_rec_neut, s_rates, r_rates,  &
     Raxis, rr, pro, qpr, dlen, det,  &    ! renaming dt-->det. In this subroutine, dt is half-step
     surfmain, surfdiv, PWI, Zmain, Zdiv, &
     rnmain, rndiv, fluxmain, fluxdiv, ymain, ydiv, depthmain, depthdiv, nmainsat, ndivsat, & 
     rcl,screen,rcmb,tsuold,dsuold,dsulold, divold, pumpold, taudiv, tauwret, leak, volpump, &
     a, b, c, d1, bet, gam, &
     Nmainret, Ndivret, rcld, rcld_refl, rcld_recl, div_flux_impl, div_fluxes_sput, &
     rclb, rcls, rclp, rclw, rclw_refl, rclw_recl, main_flux_impl, main_fluxes_sput)
     
  !
  !  Impurity transport forward modeling with default STRAHL finite-differences scheme.
  !  Refer to STRAHL manual 2018 for details of the algorithm.
  !  The recycling/pumping model, including the interaction between plasma and physical walls
  !  and the transport between the 0D neutrals reservoirs, has been widely extended wrt STRAHL,
  !  featuring a more realistic plasma-wall interaction model and a larger number of reservoirs.
  !
  ! ----------------------------------------------------------------------|

  IMPLICIT NONE

  INTEGER, INTENT(IN)       :: nion
  INTEGER, INTENT(IN)       :: ir
  INTEGER, INTENT(IN)       :: species
  REAL*8, INTENT(IN)        :: ra(ir,nion)   ! alt = old
  REAL*8, INTENT(OUT)       :: rn(ir,nion)   ! neu = new
  REAL*8, INTENT(IN)        :: diff(ir, nion)
  REAL*8, INTENT(IN)        :: conv(ir, nion)
  REAL*8, INTENT(IN)        :: par_loss_rate(ir)
  REAL*8, INTENT(IN)        :: src_prof(ir)
  REAL*8, INTENT(IN)        :: rcl_rad_prof(ir)
  REAL*8, INTENT(IN)        :: rfl_rad_prof(ir)
  REAL*8, INTENT(IN)        :: spt_rad_prof(ir,1+species)
  LOGICAL, INTENT(IN)       :: en_rec_neut
  REAL*8, INTENT(IN)        :: s_rates(ir,nion)
  REAL*8, INTENT(IN)        :: r_rates(ir,nion)
  REAL*8, INTENT(IN)        :: Raxis
  REAL*8, INTENT(IN)        :: rr(ir)
  REAL*8, INTENT(IN)        :: pro(ir)
  REAL*8, INTENT(IN)        :: qpr(ir)
  REAL*8, INTENT(IN)        :: dlen ! decay length at last radial grid point
  REAL*8, INTENT(IN)        :: det   ! full time step, named dt in subroutine `run'

  ! plasma-wall interaction inputs
  REAL*8, INTENT(IN)        :: surfmain
  REAL*8, INTENT(IN)        :: surfdiv
  LOGICAL, INTENT(IN)       :: PWI
  INTEGER, INTENT(IN)       :: Zmain
  INTEGER, INTENT(IN)       :: Zdiv
  REAL*8, INTENT(IN)        :: rnmain
  REAL*8, INTENT(IN)        :: rndiv
  REAL*8, INTENT(IN)        :: fluxmain(species)
  REAL*8, INTENT(IN)        :: fluxdiv(species)
  REAL*8, INTENT(IN)        :: ymain(1+species)
  REAL*8, INTENT(IN)        :: ydiv(1+species)
  REAL*8, INTENT(IN)        :: depthmain
  REAL*8, INTENT(IN)        :: depthdiv
  REAL*8, INTENT(IN)        :: nmainsat
  REAL*8, INTENT(IN)        :: ndivsat

  ! recycling inputs
  REAL*8, INTENT(IN)        :: rcl
  REAL*8, INTENT(IN)        :: screen
  REAL*8, INTENT(IN)        :: rcmb
  REAL*8, INTENT(INOUT)     :: tsuold   ! tsu from previous recycling step
  REAL*8, INTENT(INOUT)     :: dsuold   ! tsu from previous recycling step
  REAL*8, INTENT(INOUT)     :: dsulold  ! dsul from previous recycling step
  REAL*8, INTENT(IN)        :: divold   ! divnew from previous step (even without backflow)
  REAL*8, INTENT(IN)        :: pumpold  ! pumpnew from previous step (even without leakage)

  REAL*8, INTENT(IN)        :: taudiv
  REAL*8, INTENT(IN)        :: tauwret
  REAL*8, INTENT(IN)        :: leak
  REAL*8, INTENT(IN)        :: volpump

  ! Re-use memory allocation
  REAL*8, INTENT(INOUT)     :: a(ir,nion), b(ir,nion), c(ir,nion), d1(ir), bet(ir), gam(ir)

  REAL*8, INTENT(INOUT)     :: Nmainret
  REAL*8, INTENT(INOUT)     :: Ndivret
  REAL*8, INTENT(OUT)       :: rcld
  REAL*8, INTENT(OUT)       :: rcld_refl
  REAL*8, INTENT(OUT)       :: rcld_recl
  REAL*8, INTENT(OUT)       :: div_flux_impl
  REAL*8, INTENT(OUT)       :: div_fluxes_sput(1+species)
  REAL*8, INTENT(OUT)       :: rclb
  REAL*8, INTENT(OUT)       :: rcls
  REAL*8, INTENT(OUT)       :: rclp
  REAL*8, INTENT(OUT)       :: rclw
  REAL*8, INTENT(OUT)       :: rclw_refl
  REAL*8, INTENT(OUT)       :: rclw_recl
  REAL*8, INTENT(OUT)       :: main_flux_impl
  REAL*8, INTENT(OUT)       :: main_fluxes_sput(1+species)
  
  REAL*8 :: tauleak
  REAL*8 :: der
  INTEGER :: i, nz, s

  REAL*8 temp1, temp2, temp3, temp4, dt

  ! dr near magnetic axis
  der = rr(2) - rr(1)

  !     **************************************************
  !     ** Time CENTERED TREATMENT in radial coordinate **
  !     ** LACKNER METHOD IN Z                          **
  !     ** time step is split in two half time steps    **
  !     **************************************************
  
  ! Plasma-wall interaction
  
  if (PWI) then   ! advanced plasma-wall interaction model: wall content and fluxes determined by reflection, sputtering and wall saturation level
  
     if (rcl.ge.0) then
        rcls = (divold/taudiv)*(screen)
     else
        rcls = 0.d0
     endif
     
     call advanced_PWI_model(det, Raxis, tsuold, dsulold, dsuold, rcls, species, &
          rcmb, surfmain, surfdiv, Zmain, Zdiv, depthmain, depthdiv, nmainsat, ndivsat, &
          fluxmain, fluxdiv, rnmain, rndiv, ymain, ydiv, &
          Nmainret, Ndivret, &
          rclw, rclw_refl, rclw_recl, main_flux_impl, main_fluxes_sput, &
          rcld, rcld_refl, rcld_recl, div_flux_impl, div_fluxes_sput)
  
  else   ! simple plasma-wall interaction model: wall content and fluxes determined by simple recl. coeff. and retention time
  
     if (rcl.ge.0) then   ! recycling activated --> fluxes rclw, rcld present
  
        rclw = rcl*(tsuold+dsulold)   ! OLD flux towards main wall which can be recycled
        rcld = rcl*(1.-rcmb)*(dsuold+rcls)   ! OLD flux towards divertor wall which can be recycled
     
        if (tauwret.gt.0.0d0) then
           ! include flux from particles previously retained at the main wall
           Nmainret = Nmainret * (1-det/tauwret) + rclw*det   ! number of particles temporarily retained at main wall
           Ndivret = Ndivret * (1-det/tauwret) + rcld*det     ! number of particles temporarily retained at divertor wall
           rclw = Nmainret/tauwret   ! total recycled flux from main wall which will be a NEW source for the core plasma
           rcld = Ndivret/tauwret   ! total recycled flux from divertor wall which will be a NEW source for the divertor neutrals reservoir
        
        endif
     
     else   ! recycling de-activated --> no fluxes rclw, rcld, and temporary wall reservoirs are not filled
  
        rcld = 0.d0
        rclw = 0.d0
 
     endif
     
     rclw_refl = 0.d0
     rcld_refl = 0.d0
     rclw_recl = 0.d0
     rcld_recl = 0.d0
     main_flux_impl = 0.d0
     div_flux_impl = 0.d0
     do s = 1,1+species
        main_fluxes_sput(s) = 0.d0
        div_fluxes_sput(s) = 0.d0
     enddo
  
  endif
  
  ! Backflow from divertor
  
  if (rcl.ge.0) then   ! activated divertor/pump return --> fluxes rclb, rcls, rclp present
  
     rclb = (divold/taudiv)*(1-screen) ! non-screened backflow from previous time step
     rcls = (divold/taudiv)*(screen) ! screened backflow from previous time step
     if (leak.gt.0) then    ! pump reservoir present and activated leakage
        tauleak = volpump/leak
        rclp = pumpold/tauleak
     else
        rclp = 0.d0
     endif
     
  else   ! no divertor/pump return --> no fluxes rclb, rcls, rclp at all
  
     rclb = 0.d0
     rcls = 0.d0
     rclp = 0.d0
 
  endif

  ! sum radial sources from external source time history and recycling
  ! The radial profile of externally injected neutrals and recycled neutrals can be different,
  !   depending on the energy set to the recycling neutrals (and to the calculated reflection
  !   and sputtering energies, if the advanced PWI model is used)
  
  if (en_rec_neut) then
     
     if (PWI) then   ! Advanced PWI model used --> reflected and sputtered neutrals are energetic
        rn(:,1) = src_prof   ! Add contribute of externally injected neutrals
        rn(:,1) = rn(:,1) + rfl_rad_prof*(rclw_refl)   ! Add contribute of energetic reflected neutrals
        rn(:,1) = rn(:,1) + rcl_rad_prof*(rclw_recl)   ! Add contribute of thermally recycled neutrals
        do s = 1,1+species
           rn(:,1) = rn(:,1) + spt_rad_prof(:,s)*(main_fluxes_sput(s))   ! Add contribute of energetic sputtered neutrals
        enddo
        rn(:,1) = rn(:,1) + rcl_rad_prof*(rclp + rclb)   ! Add contribute of thermal neutrals returned/leaked from divertor

     else   ! Simple PWI model used --> recycled neutrals will be thermal regardless of en_rec_neut
        rn(:,1) = src_prof + rcl_rad_prof*(rclw + rclp + rclb)

     endif
  
  else   ! Recycled neutrals (also the reflected and sputtered ones) are always thermal regardless of the used PWI model
  
     rn(:,1) = src_prof + rcl_rad_prof*(rclw + rclp + rclb)
  
  endif
  
  dt = det/2.
  !     ********** first half time step direction up ********

  do nz=2,nion
     !     r=0
     a(1,nz)=0.
     c(1,nz)=-2.*dt*diff(1,nz)*pro(1)
     b(1,nz)=1.-c(1,nz)+2.*dt*conv(2,nz)/der
     d1(1)=ra(1,nz)*(2.-b(1,nz))-ra(2,nz)*c(1,nz)

     !     r=r or at r+db respectively

     if (dlen > 0.d0) then
        temp1=4.*dt*pro(ir)**2*diff(ir,nz)
        temp2=.5*dt*(qpr(ir)*diff(ir,nz)-pro(ir)*conv(ir,nz))
        temp3=.5*dt*(par_loss_rate(ir)+conv(ir,nz)/rr(ir))
        temp4=1./pro(ir)/dlen

        a(ir,nz)=-temp1
        b(ir,nz)=1.+(1.+.5*temp4)*temp1+temp4*temp2+temp3
        c(ir,nz)=0.d0
        d1(ir)=-ra(ir-1,nz)*a(ir,nz)+ra(ir,nz)*(2.-b(ir,nz))
        b(ir,nz)=b(ir,nz)+dt*s_rates(ir,nz)
        d1(ir)=d1(ir)-dt*( ra(ir,nz)*r_rates(ir,nz-1) -rn(ir,nz-1)*s_rates(ir,nz-1))
        if (nz < nion) d1(ir)=d1(ir)+dt*ra(ir,nz+1)*r_rates(ir,nz)
     end if
     if (dlen <= 0.d0) then !Edge Conditions for rn(ir)=0.
        a(ir,nz)=0
        b(ir,nz)=1.
        c(ir,nz)= 0.
        d1(ir)=ra(ir,nz)
     end if

     !     normal coefficients

     do i=2,ir-1

        temp1=dt*pro(i)**2
        temp2=4.*temp1*diff(i,nz)
        temp3=dt/2.*(par_loss_rate(i)+pro(i)*(conv(i+1,nz)-conv(i-1,nz))+ conv(i,nz)/rr(i))
        a(i,nz)=.5*dt*qpr(i)*diff(i,nz)+temp1* (.5*(diff(i+1,nz)-diff(i-1,nz)  &
             -conv(i,nz)/pro(i))-2.*diff(i,nz))
        b(i,nz)=1.+temp2+temp3
        c(i,nz)=-temp2-a(i,nz)
        d1(i)=-ra(i-1,nz)*a(i,nz)+ra(i,nz)*(2.-b(i,nz)) -ra(i+1,nz)*c(i,nz)
     end do

     do i=1,ir-1
        b(i,nz)=b(i,nz)+dt*s_rates(i,nz)
        d1(i)=d1(i)-dt*(ra(i,nz)*r_rates(i,nz-1) -rn(i,nz-1)*s_rates(i,nz-1))   !ra --> "alt" (i-1) density; rn --> "current" (i) density
        if (nz < nion) d1(i)=d1(i)+dt*ra(i,nz+1)*r_rates(i,nz)   !r_rates --> recomb; s_rates --> ioniz
     end do

     !     solution of tridiagonal equation system

     bet(1)=b(1,nz)
     gam(1)=d1(1)/b(1,nz)
     do i=2,ir
        bet(i)=b(i,nz)-(a(i,nz)*c(i-1,nz))/bet(i-1)
        gam(i)=(d1(i)-a(i,nz)*gam(i-1))/bet(i)
     end do
     rn(ir,nz)=gam(ir)

     do i=ir-1,1,-1
        rn(i,nz)=gam(i)-(c(i,nz)*rn(i+1,nz))/bet(i)
     end do

  end do

  !     ********** second half time step Z-direction down ********

  do nz=nion,2,-1

     !     r=0
     a(1,nz)=0.
     c(1,nz)=-2.*dt*diff(1,nz)*pro(1)
     b(1,nz)=1.-c(1,nz)+2.*dt*conv(2,nz)/der
     d1(1)=rn(1,nz)*(2.-b(1,nz))-rn(2,nz)*c(1,nz)

     !     r=r or at r+db respectively
     if (dlen > 0.d0) then
        temp1=4.*dt*pro(ir)**2*diff(ir,nz)
        temp2=.5*dt*(qpr(ir)*diff(ir,nz)-pro(ir)*conv(ir,nz))
        temp3=.5*dt*(par_loss_rate(ir)+conv(ir,nz)/rr(ir))
        temp4=1./pro(ir)/dlen

        a(ir,nz)=-temp1
        b(ir,nz)=1.+(1.+.5*temp4)*temp1+temp4*temp2+temp3
        c(ir,nz)=0.d0
        d1(ir)=-rn(ir-1,nz)*a(ir,nz)+rn(ir,nz)*(2.-b(ir,nz))
        b(ir,nz)=b(ir,nz)+dt*r_rates(ir,nz-1)
        d1(ir)=d1(ir)-dt*(rn(ir,nz)*s_rates(ir,nz) -rn(ir,nz-1)*s_rates(ir,nz-1))
        if (nz < nion) d1(ir)=d1(ir)+dt*rn(ir,nz+1)*r_rates(ir,nz)  !!!!
     end if
     if (dlen <= 0.d0) then !Edge Conditions for rn(ir)=0.
        a(ir,nz)=0
        b(ir,nz)=1.
        c(ir,nz)= 0.
        d1(ir)=rn(ir,nz)
     end if

     !     normal coefficients

     do i=2,ir-1
        temp1=dt*pro(i)**2
        temp2=4.*temp1*diff(i,nz)
        temp3=dt/2.*(par_loss_rate(i)+pro(i)*(conv(i+1,nz)-conv(i-1,nz))+ conv(i,nz)/rr(i))

        a(i,nz)=.5*dt*qpr(i)*diff(i,nz)+temp1* (.5*(diff(i+1,nz)-diff(i-1,nz)  &
             -conv(i,nz)/pro(i))-2.*diff(i,nz))
        b(i,nz)=1.+temp2+temp3
        c(i,nz)=-temp2-a(i,nz)
        d1(i)=-rn(i-1,nz)*a(i,nz)+rn(i,nz)*(2.-b(i,nz)) -rn(i+1,nz)*c(i,nz)
     end do

     do i=1,ir-1
        b(i,nz)=b(i,nz)+dt*r_rates(i,nz-1)
        d1(i)=d1(i)-dt*(rn(i,nz)*s_rates(i,nz)- rn(i,nz-1)*s_rates(i,nz-1))
        if (nz < nion) d1(i)=d1(i)+dt*rn(i,nz+1)*r_rates(i,nz)   !!!!!
     end do

     !     solution of tridiagonal equation system

     bet(1)=b(1,nz)
     gam(1)=d1(1)/b(1,nz)
     do i=2,ir
        bet(i)=b(i,nz)-(a(i,nz)*c(i-1,nz))/bet(i-1)
        gam(i)=(d1(i)-a(i,nz)*gam(i-1))/bet(i)
     end do
     rn(ir,nz)=gam(ir)    !!!!

     do i=ir-1,1,-1
        rn(i,nz)=gam(i)-(c(i,nz)*rn(i+1,nz))/bet(i)     !!!!
     end do
  end do


  return
end subroutine impden0





subroutine impden1(nion, ir, species, ra, rn, diff, conv, par_loss_rate, &
     src_prof, rcl_rad_prof, rfl_rad_prof, spt_rad_prof, en_rec_neut, s_rates, r_rates,  &
     Raxis, rr, fall_outsol, det,  &    ! renaming dt-->det. In this subroutine, dt is half-step
     surfmain, surfdiv, PWI, Zmain, Zdiv, &
     rnmain, rndiv, fluxmain, fluxdiv, ymain, ydiv, depthmain, depthdiv, nmainsat, ndivsat, & 
     rcl, screen, rcmb, tsuold, dsuold, dsulold, divold, pumpold, taudiv, tauwret, leak, volpump, &
     evolveneut, Nmainret, Ndivret, rcld, rcld_refl, rcld_recl, div_flux_impl, div_fluxes_sput, &
     rclb, rcls, rclp, rclw, rclw_refl, rclw_recl, main_flux_impl, main_fluxes_sput)
  !
  !  Impurity transport forward modeling with Linder's finite-volume scheme.
  !  See Linder et al. NF 2020
  !  The recycling/pumping model is however the same as in impden0.
  !
  ! ----------------------------------------------------------------------|
  implicit none
    
  INTEGER, INTENT(IN)       :: nion
  INTEGER, INTENT(IN)       :: ir
  INTEGER, INTENT(IN)       :: species
  REAL*8, INTENT(IN)        :: ra(ir,nion)   ! old imp density
  REAL*8, INTENT(OUT)       :: rn(ir,nion)   ! new imp density
  REAL*8, INTENT(IN)        :: diff(ir, nion)
  REAL*8, INTENT(IN)        :: conv(ir, nion)
  REAL*8, INTENT(IN)        :: par_loss_rate(ir)
  REAL*8, INTENT(IN)        :: src_prof(ir)
  REAL*8, INTENT(IN)        :: rcl_rad_prof(ir)
  REAL*8, INTENT(IN)        :: rfl_rad_prof(ir)
  REAL*8, INTENT(IN)        :: spt_rad_prof(ir,1+species)
  LOGICAL, INTENT(IN)       :: en_rec_neut
  REAL*8, INTENT(IN)        :: s_rates(ir,nion)    ! ionization
  REAL*8, INTENT(IN)        :: r_rates(ir,nion)   ! recombination
  REAL*8, INTENT(IN)        :: Raxis
  REAL*8, INTENT(IN)        :: rr(ir)
  REAL*8, INTENT(IN)        :: fall_outsol  !dlen ! decay length at last radial grid point
  REAL*8, INTENT(IN)        :: det   ! full time step, named dt in subroutine `run'

  ! plasma-wall interaction inputs
  REAL*8, INTENT(IN)        :: surfmain
  REAL*8, INTENT(IN)        :: surfdiv
  LOGICAL, INTENT(IN)       :: PWI
  INTEGER, INTENT(IN)       :: Zmain
  INTEGER, INTENT(IN)       :: Zdiv
  REAL*8, INTENT(IN)        :: rnmain
  REAL*8, INTENT(IN)        :: rndiv
  REAL*8, INTENT(IN)        :: fluxmain(species)
  REAL*8, INTENT(IN)        :: fluxdiv(species)
  REAL*8, INTENT(IN)        :: ymain(1+species)
  REAL*8, INTENT(IN)        :: ydiv(1+species)
  REAL*8, INTENT(IN)        :: depthmain
  REAL*8, INTENT(IN)        :: depthdiv
  REAL*8, INTENT(IN)        :: nmainsat
  REAL*8, INTENT(IN)        :: ndivsat

  ! recycling inputs
  REAL*8, INTENT(IN)        :: rcl
  REAL*8, INTENT(IN)        :: screen
  REAL*8, INTENT(IN)        :: rcmb
  REAL*8, INTENT(INOUT)     :: tsuold   ! tsu from previous recycling step
  REAL*8, INTENT(INOUT)     :: dsuold   ! dsu from previous recycling step
  REAL*8, INTENT(INOUT)     :: dsulold  ! dsul from previous recycling step
  REAL*8, INTENT(IN)        :: divold   ! divnew from previous step (even without backflow)
  REAL*8, INTENT(IN)        :: pumpold  ! pumpnew from previous step (even without leakage)

  REAL*8, INTENT(IN)        :: taudiv
  REAL*8, INTENT(IN)        :: tauwret
  REAL*8, INTENT(IN)        :: leak
  REAL*8, INTENT(IN)        :: volpump

  ! Extras
  LOGICAL, INTENT(IN)       :: evolveneut
  
  REAL*8, INTENT(INOUT)     :: Nmainret
  REAL*8, INTENT(INOUT)     :: Ndivret
  REAL*8, INTENT(OUT)       :: rcld
  REAL*8, INTENT(OUT)       :: rcld_refl
  REAL*8, INTENT(OUT)       :: rcld_recl
  REAL*8, INTENT(OUT)       :: div_flux_impl
  REAL*8, INTENT(OUT)       :: div_fluxes_sput(1+species)
  REAL*8, INTENT(OUT)       :: rclb
  REAL*8, INTENT(OUT)       :: rcls
  REAL*8, INTENT(OUT)       :: rclp
  REAL*8, INTENT(OUT)       :: rclw
  REAL*8, INTENT(OUT)       :: rclw_refl
  REAL*8, INTENT(OUT)       :: rclw_recl
  REAL*8, INTENT(OUT)       :: main_flux_impl
  REAL*8, INTENT(OUT)       :: main_fluxes_sput(1+species)

  REAL*8 :: src(ir), tauleak
  INTEGER :: i, nz, ns, s

  REAL*8 :: dt

  ! nt is the solution of the TDMA
  real*8 ::  nt(ir), a(ir), b(ir), c(ir), d(ir)
  
  ! Lackner method: split time step in 2 half time steps
  dt = det/2.    
  
  ! Plasma-wall interaction
  
  if (PWI) then   ! advanced plasma-wall interaction model: wall content and fluxes determined by reflection, sputtering and wall saturation level
  
     if (rcl.ge.0) then
        rcls = (divold/taudiv)*(screen)
     else
        rcls = 0.d0
     endif
     
     call advanced_PWI_model(det, Raxis, tsuold, dsulold, dsuold, rcls, species, &
          rcmb, surfmain, surfdiv, Zmain, Zdiv, depthmain, depthdiv, nmainsat, ndivsat, &
          fluxmain, fluxdiv, rnmain, rndiv, ymain, ydiv, &
          Nmainret, Ndivret, &
          rclw, rclw_refl, rclw_recl, main_flux_impl, main_fluxes_sput, &
          rcld, rcld_refl, rcld_recl, div_flux_impl, div_fluxes_sput)
  
  else   ! simple plasma-wall interaction model: wall content and fluxes determined by simple recl. coeff. and retention time
  
     if (rcl.ge.0) then   ! recycling activated --> fluxes rclw, rcld present
  
        rclw = rcl*(tsuold+dsulold)   ! OLD flux towards main wall which can be recycled
        rcld = rcl*(1.-rcmb)*(dsuold+rcls)   ! OLD flux towards divertor wall which can be recycled
     
        if (tauwret.gt.0.0d0) then
           ! include flux from particles previously retained at the main wall
           Nmainret = Nmainret * (1-det/tauwret) + rclw*det   ! number of particles temporarily retained at main wall
           Ndivret = Ndivret * (1-det/tauwret) + rcld*det     ! number of particles temporarily retained at divertor wall
           rclw = Nmainret/tauwret   ! total recycled flux from main wall which will be a NEW source for the core plasma
           rcld = Ndivret/tauwret   ! total recycled flux from divertor wall which will be a NEW source for the divertor neutrals reservoir
        
        endif
     
     else   ! recycling de-activated --> no fluxes rclw, rcld, and temporary wall reservoirs are not filled
  
        rcld = 0.d0
        rclw = 0.d0
 
     endif
     
     rclw_refl = 0.d0
     rcld_refl = 0.d0
     rclw_recl = 0.d0
     rcld_recl = 0.d0
     main_flux_impl = 0.d0
     div_flux_impl = 0.d0
     do s = 1,1+species
        main_fluxes_sput(s) = 0.d0
        div_fluxes_sput(s) = 0.d0
     enddo
  
  endif
  
  ! Backflow from divertor
  
  if (rcl.ge.0) then   ! activated divertor/pump return --> fluxes rclb, rcls, rclp present
  
     rclb = (divold/taudiv)*(1-screen) ! non-screened backflow from previous time step
     rcls = (divold/taudiv)*(screen) ! screened backflow from previous time step
     if (leak.gt.0) then    ! pump reservoir present and activated leakage
        tauleak = volpump/leak
        rclp = pumpold/tauleak
     else
        rclp = 0.d0
     endif
     
  else   ! no divertor/pump return --> no fluxes rclb, rcls, rclp at all
  
     rclb = 0.d0
     rcls = 0.d0
     rclp = 0.d0
 
  endif
  
  ! sum radial sources from external source time history and recycling
  ! The radial profile of externally injected neutrals and recycled neutrals can be different,
  !   depending on the energy set to the recycling neutrals (and to the calculated reflection
  !   and sputtering energies, if the advanced PWI model is used)

  if (en_rec_neut) then
     
     if (PWI) then   ! Advanced PWI model used --> reflected and sputtered neutrals are energetic
        src = src_prof   ! Add contribute of externally injected neutrals
        src = src + rfl_rad_prof*(rclw_refl)   ! Add contribute of energetic reflected neutrals
        src = src + rcl_rad_prof*(rclw_recl)   ! Add contribute of thermally recycled neutrals
        do s = 1,1+species
           src = src + spt_rad_prof(:,s)*(main_fluxes_sput(s))   ! Add contribute of energetic sputtered neutrals
        enddo
        src = src + rcl_rad_prof*(rclp + rclb)   ! Add contribute of thermal neutrals returned/leaked from divertor

     else   ! Simple PWI model used --> recycled neutrals will be thermal regardless of en_rec_neut
        src = src_prof + rcl_rad_prof*(rclw + rclp + rclb)

     endif

  else   ! Recycled neutrals (also the reflected and sputtered ones) are always thermal regardless of the used PWI model

     src = src_prof + rcl_rad_prof*(rclw + rclp + rclb)

  endif 
  
  ! select whether neutrals should be evolved
  ns = 2
  if (evolveneut) ns = 1


  ! ----- First half time step: direction up --------------------|
  ! Externally provided neutrals
  if (ns.eq.1) then
     call impden_constTranspMatrix(ra(:,1), diff(:,1), conv(:,1),&    !na(1), nd(1), nv(1),&
          ir, dt, fall_outsol, par_loss_rate, rr, a, b, c, d)

     do i=1,ir
        b(i)    = b(i) + dt*s_rates(i,1)
        d(i)    = d(i) + dt*src(i)
     enddo
     call TDMA(a, b, c, d, ir, nt)
  else
     nt = 0.d0
     ! radial profile of neutrals (given as input)
     do i=1,ir
        rn(i,1) = src(i)
     end do
  endif
  
  ! Ions and recombined neutrals
  do nz=ns,nion             
     ! Construct transport matrix
     call impden_constTranspMatrix(ra(:,nz), diff(:,nz),&
          conv(:,nz), ir, dt, fall_outsol, par_loss_rate, rr, a, b, c, d)
     
     ! Add ionization and recombination
     do i=1,ir
        b(i)    = b(i) + dt*s_rates(i,nz)
        if (nz.gt.1)&
             d(i)    = d(i) + dt*(rn(i,nz-1)*s_rates(i,nz-1) - ra(i,nz)*r_rates(i,nz-1))
        if (nz.lt.nion)&
             d(i)    = d(i) + dt*ra(i,nz+1)*r_rates(i,nz)
        if (nz.eq.2)&
             d(i)    = d(i) + dt*nt(i)*s_rates(i,1)
     enddo
     
     ! Solve tridiagonal system of equations
     call TDMA(a, b, c, d, ir, rn(:,nz))
  enddo
  
  ! ----- Second half time step: direction down -----------------|
  ! Ions and recombined neutrals
  do nz=nion,ns,-1
     ! Construct transport matrix
     call impden_constTranspMatrix(rn(:,nz), diff(:,nz),&
          conv(:,nz), ir, dt, fall_outsol, par_loss_rate, rr, a, b, c, d)
     
     ! Add ionization and recombination
     do i=1,ir
        d(i)    = d(i) - dt*rn(i,nz)*s_rates(i,nz)
        if (nz.gt.1) then
           b(i)    = b(i) + dt*r_rates(i,nz-1)
           d(i)    = d(i) + dt*rn(i,nz-1)*s_rates(i,nz-1)
        endif
        if (nz.lt.nion) &
             d(i)    = d(i) + dt*rn(i,nz+1)*r_rates(i,nz) 
        if (nz.eq.2) &
             d(i)    = d(i) + dt*nt(i)*s_rates(i,1)
     enddo
     
     ! Solve tridiagonal equation system
     call TDMA(a, b, c, d, ir, rn(:,nz))
  enddo
  
  ! Externally provided neutrals
  if (ns.eq.1) then
     call impden_constTranspMatrix(nt, diff(:,1), conv(:,1), ir,&
          dt, fall_outsol, par_loss_rate, rr, a, b, c, d)
          
     do i=1,ir
        d(i)    = d(i) - dt*nt(i)*s_rates(i,1) + dt*src(i)
     enddo
     call TDMA(a, b, c, d, ir, rn(:,1))
  endif
    
  ! ----- Calculate ionization rate -----------------------------|
  ! Bulk ions & neutrals
  ! do nz=1,nion-1
  !    do i=1,ir
  !       ioniz_loss(i,nz) = s(i,nz)*rn(i,nz)
  !    enddo
  ! enddo
  ! ! External neutrals
  ! do i=1,ir
  !    ioniz_loss(i,1) = ioniz_loss(i,1) + nt(i)*s_rates(i,1)
  ! enddo
  
  return
end subroutine impden1
!======================================================================|





!======================================================================|
subroutine impden_constTranspMatrix(rnt, dimp, vimp, ir, dt, &
     fall_outsol, par_loss_rate, rr, a, b, c, d)
  ! ----------------------------------------------------------------------|
  !     Construct transport matrix for charge state IZ of impurity 
  !     species IM.
  ! ----------------------------------------------------------------------|
  implicit none
  
  ! ----- Global variables --------------------------------------|
  integer, intent(in) ::  ir
  
  real*8, intent(in) :: rnt(ir), dt, fall_outsol, dimp(ir), vimp(ir),&
       par_loss_rate(ir), rr(ir)
  
  real*8, intent(out) ::  a(ir), b(ir), c(ir), d(ir)
  
  ! ----- Local variables ---------------------------------------|
  real*8 :: coefm, coefp, Dm, Dp, drm, drp, gi, kam, kap, rm, rp, vm, vp
  
  integer :: i
  
  ! ----------------------------------------------------------------------|
  !     Temporal component
  ! ----------------------------------------------------------------------|
  a(:) = 0
  b(:) = 1
  c(:) = 0
  d(:) = rnt(:)
  
  ! ----------------------------------------------------------------------|
  !   Central point (r = 0)
  !   Enforces dn/dr|_r=0 = 0 and v(r=0) = 0
  ! ----------------------------------------------------------------------|
  ! ----- Geometric factors -------------------------------------|
  drp   = rr(2)-rr(1)
  rp    = .5*(rr(1)+rr(2))
  gi    = 2./(rp**2)
  
  ! ----- Diffusion contribution --------------------------------|
  Dp    = .5*(dimp(1)+dimp(2))
  coefp = .5*dt*gi*rp*Dp/drp
  
  a(1)  = a(1)
  b(1)  = b(1) + coefp
  c(1)  = c(1) - coefp
  d(1)  = d(1) + coefp*(rnt(2) - rnt(1))
  
  ! ----- Advection contribution --------------------------------|
  vp    = .5*(vimp(1)+vimp(2))
  kap   = 0.
  if (abs(vp).gt.2.*Dp/drp) kap = sign(max(0.d0, 1.-2.*Dp/(drp*abs(vp))), vp)
  
  coefp = .25*dt*gi*rp*vp
  
  a(1)  = a(1)
  b(1)  = b(1) + coefp*(1+kap)
  c(1)  = c(1) + coefp*(1-kap)
  d(1)  = d(1) - coefp*((1+kap)*rnt(1) + (1-kap)*rnt(2))
  
  ! ----------------------------------------------------------------------|
  !     Interior points
  ! ----------------------------------------------------------------------|
  do i=2,ir-1
     ! ----- Geometric factors -------------------------------------|
     drm     = drp
     drp     = rr(i+1)-rr(i)
     rm      = rp
     rp      = .5*(rr(i)+rr(i+1))
     gi      = 2./(rp**2-rm**2)
     
     ! ----- Diffusion contribution --------------------------------|
     Dm      = Dp
     Dp      = .5*(dimp(i)+dimp(i+1))
     coefm   = .5*dt*gi*rm*Dm/drm
     coefp   = .5*dt*gi*rp*Dp/drp
     
     a(i)    = a(i) - coefm
     b(i)    = b(i) + coefm + coefp
     c(i)    = c(i)         - coefp
     d(i)    = d(i) - coefm*(rnt(i  ) - rnt(i-1))&
                    + coefp*(rnt(i+1) - rnt(i  ))
     
     ! ----- Advection contribution --------------------------------|
     vm      = vp
     vp      = .5*(vimp(i)+vimp(i+1))
     
     kam     = 0.
     if (abs(vm).gt.2.*Dm/drm)&
          kam     = sign(max(0.d0, 1.-2.*Dm/(drm*abs(vm))), vm)
     kap     = 0.
     if (abs(vp).gt.2.*Dp/drp)&
          kap     = sign(max(0.d0, 1.-2.*Dp/(drp*abs(vp))), vp)
     
     coefm   = .25*dt*gi*rm*vm
     coefp   = .25*dt*gi*rp*vp
     
     a(i)    = a(i) - coefm*(1+kam)
     b(i)    = b(i) - coefm*(1-kam) + coefp*(1+kap)
     c(i)    = c(i)                 + coefp*(1-kap)
     d(i)    = d(i) + coefm*((1+kam)*rnt(i-1) + (1-kam)*rnt(i  ))&
                    - coefp*((1+kap)*rnt(i  ) + (1-kap)*rnt(i+1))

  enddo
  
  !----------------------------------------------------------------------|
  !     Outer point (r = r)
  !----------------------------------------------------------------------|
  ! ----- Geometric factors -------------------------------------|
  drm       = drp
  rm        = rp
  rp        = rr(ir) + .5*drm
  gi        = 2./(rp**2-rm**2)
  
  ! ----- Diffusion contribution --------------------------------|
  Dm        = Dp
  Dp        = dimp(ir)
  coefm     = .5*dt*gi*rm*Dm/drm
  coefp     = .5*dt*gi*rp*Dp
  
  a(ir)     = a(ir)- coefm
  b(ir)     = b(ir)+ coefm 
  d(ir)     = d(ir)- coefm*(rnt(ir) - rnt(ir-1))
  
  if (fall_outsol.gt.0.d0) then
     b(ir)   = b(ir)+ coefp/fall_outsol
     d(ir)   = d(ir)- coefp/fall_outsol*rnt(ir)
  endif
  
  ! ----- Advection contribution --------------------------------|
  vm        = vp
  vp        = max(0.d0, vimp(ir))
  
  kam       = 0.
  if (abs(vm).gt.2.*Dm/drm)&
       kam = sign(max(0.d0, 1.d0-2.d0*Dm/(drm*abs(vm))), vm)
  kap       = 0.
  if (abs(vp).gt.2.*Dp/drp)&
       kap = sign(max(0.d0, 1.d0-2.d0*Dp/(drp*abs(vp))), vp)
  
  coefm     = .25*dt*gi*rm*vm
  coefp     = .25*dt*gi*rp*vp
  
  a(ir)     = a(ir)- coefm*(1+kam)
  b(ir)     = b(ir)- coefm*(1-kam) 
  d(ir)     = d(ir)+ coefm*((1+kam)*rnt(ir-1) + (1-kam)*rnt(ir))
  
  if (fall_outsol.gt.0.d0) then
     b(ir)   = b(ir)+ coefp*(2-(1-kap)*drp/fall_outsol)
     d(ir)   = d(ir)- coefp*(2-(1-kap)*drp/fall_outsol)*rnt(ir)
  endif
  
  !----------------------------------------------------------------------|
  ! SOL Losses
  !----------------------------------------------------------------------|
  ! ----- Interior points ---------------------------------------|
  do i=2,ir-1
     coefm   = .5*dt*par_loss_rate(i)
     
     b(i)    = b(i) +  coefm
     d(i)    = d(i) -  coefm*rnt(i)
  enddo
  
  ! ----- Outer point (r = r) -----------------------------------|
  if (fall_outsol.gt.0.d0) then
     coefm   = .5*dt*par_loss_rate(ir)
     
     b(i)    = b(i) +  coefm
     d(i)    = d(i) -  coefm*rnt(ir)
  endif
  
  return
end subroutine impden_constTranspMatrix






!======================================================================|
subroutine TDMA(dLow, dMain, dUpp, rhs, n, sol)
  ! ---------------------------------------------------------------------|
  !     TriDiagonal Matrix Algorithm:
  !     Solves linear system of equations if coefficient matrix M is in a
  !     tridiagonal form. 
  !     See: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
  !
  !     Input:
  !       dLow        Lower diagonal M(i,i-1). Starts with element 
  !                   M(1,n) = 0.
  !       dMain       Main diagonal M(i,i).
  !       dUpp        Upper diagnonal M(i,i+1). Ends with element 
  !                   M(n,1) = 0.
  !       rhs         Right-hand-side of system of equations
  !       n           Number of equations.
  !     Output:
  !       sol         Solution of system of equations.
  ! ---------------------------------------------------------------------|
  implicit none
  
  ! ----- I/O variables -----------------------------------------|
  real*8, dimension(n), intent(in) :: dLow, dMain, dUpp, rhs
  integer, intent(in) :: n
  
  real*8, dimension(n), intent(out) ::  sol 
  
  ! ----- Local variables ---------------------------------------|
  integer :: i
  
  real*8, dimension(n) :: rhsP, dMainP
  
  
  ! ----- Modify coefficients -----------------------------------|
  dMainP(1) = dMain(1); rhsP(1) = rhs(1)
  do i=2,n
     dMainP(i) = dMain(i) - dLow(i)*dUpp(i-1)/dMainP(i-1)
     rhsP(i) = rhs(i) - dLow(i)*rhsP(i-1)/dMainP(i-1) 
  enddo
  
  ! ----- Construct solution ------------------------------------|
  sol(n) = rhsP(n)/dMainP(n)  
  do i=n-1,1,-1
     sol(i) = (rhsP(i) - dUpp(i)*sol(i+1))/dMainP(i)
  enddo

  return
end subroutine TDMA






!============================================================================================|
subroutine advanced_PWI_model(det, Raxis, tsuold, dsulold, dsuold, rcls, species, &
     rcmb, surfmain, surfdiv, Zmain, Zdiv, depthmain, depthdiv, nmainsat, ndivsat, &
     fluxmain, fluxdiv, rnmain, rndiv, ymain, ydiv, &
     Nmainret, Ndivret, &
     rclw, rclw_refl, rclw_recl, main_flux_impl, main_fluxes_sput, &
     rcld, rcld_refl, rcld_recl, div_flux_impl, div_fluxes_sput)
  ! -----------------------------------------------------------------------------------------|
  !  Apply the advanced plasma-wall interaction model for the calculation of
  !    reflected, recycled and sputtered fluxes at the walls
  ! -----------------------------------------------------------------------------------------|
  implicit none
  
  ! ----- I/O variables ---------------------------------------------------------------------|
  
  real*8,  intent(in)     :: det
  real*8,  intent(in)     :: Raxis
  real*8,  intent(inout)  :: tsuold
  real*8,  intent(inout)  :: dsulold
  real*8,  intent(inout)  :: dsuold 
  real*8,  intent(inout)  :: rcls
  integer, intent(in)     :: species
  
  real*8,  intent(in)     :: rcmb
  real*8,  intent(in)     :: surfmain
  real*8,  intent(in)     :: surfdiv
  integer, intent(in)     :: Zmain
  integer, intent(in)     :: Zdiv
  real*8,  intent(in)     :: depthmain
  real*8,  intent(in)     :: depthdiv
  real*8,  intent(in)     :: nmainsat
  real*8,  intent(in)     :: ndivsat
  
  real*8,  intent(in)     :: fluxmain(species)
  real*8,  intent(in)     :: fluxdiv(species)
  real*8,  intent(in)     :: rnmain
  real*8,  intent(in)     :: rndiv
  real*8,  intent(in)     :: ymain(1+species)
  real*8,  intent(in)     :: ydiv(1+species)
  
  real*8,  intent(inout)  :: Nmainret 
  real*8,  intent(inout)  :: Ndivret
  
  real*8,  intent(inout)  :: rclw 
  real*8,  intent(inout)  :: rclw_refl
  real*8,  intent(inout)  :: rclw_recl
  real*8,  intent(inout)  :: main_flux_impl
  real*8,  intent(inout)  :: main_fluxes_sput(1+species)
  
  real*8,  intent(inout)  :: rcld 
  real*8,  intent(inout)  :: rcld_refl
  real*8,  intent(inout)  :: rcld_recl
  real*8,  intent(inout)  :: div_flux_impl
  real*8,  intent(inout)  :: div_fluxes_sput(1+species)
  
  ! ----- Local variables -------------------------------------------------------------------|
  real*8     :: circ
  real*8     :: sigmamainret
  real*8     :: sigmadivret
  real*8     :: mainsatlevel
  real*8     :: divsatlevel
  real*8     :: rclw_sput
  real*8     :: rcld_sput
  real*8     :: Cmain
  real*8     :: Cdiv
  integer    :: s
  
  circ = 2.0 * 3.14159 * Raxis
    
  ! convert old fluxes and content into absolute quantities
  tsuold = tsuold * circ   ! OLD radial edge flux to main wall in s^-1
  dsulold = dsulold * circ   ! OLD parallel limiter flux to main wall in s^-1
  dsuold = dsuold * circ   ! OLD parallel loss to divertor wall in s^-1
  rcls = rcls * circ   ! OLD screened divertor backflow from divertor wall in s^-1
  Nmainret = Nmainret * circ   ! OLD absolute number of particles retained at the main wall
  Ndivret = Ndivret * circ   ! OLD absolute number of particles retained at the main wall

  rclw_refl = (tsuold+dsulold) * rnmain   ! reflected impurity flux from main wall in s^-1 (NEW source for core plasma)
  rcld_refl = (1.-rcmb)*(dsuold+rcls) * rndiv   ! reflected impurity flux from divertor wall in s^-1 (NEW source for divertor reservoir)
 
  sigmamainret = (Nmainret / surfmain) * 1.0E4   ! OLD surface implantation density at the main wall in m^-2
  sigmadivret = (Ndivret / surfdiv) * 1.0E4   ! OLD surface implantation density at the divertor wall in m^-2
 
  mainsatlevel = sigmamainret / nmainsat   ! OLD saturation level of the main wall
  divsatlevel = sigmadivret / ndivsat   ! OLD saturation level of the main wall
 
  rclw_recl = (tsuold+dsulold) * (1.0 - rnmain) * mainsatlevel   ! promptly recycled flux into the main wall in s^-1 (NEW source for core plasma)
  rcld_recl = (1.-rcmb)*(dsuold+rcls) * (1.0 - rndiv) * divsatlevel   ! promptly recycled flux into the divertor wall in s^-1 (NEW source for core plasma)
 
  main_flux_impl = (tsuold+dsulold) * (1.0 - rnmain) * (1.0 - mainsatlevel)   ! implanted flux into the main wall in s^-1
  div_flux_impl = (1.-rcmb)*(dsuold+rcls) * (1.0 - rndiv) * (1.0 - divsatlevel)   ! implanted flux into the divertor wall in s^-1
 
  ! Convert OLD surface implantation densities, in m^-2, into concentrations
  call wall_density_to_concentration(sigmamainret, Zmain, depthmain, Cmain)   ! OLD impurity concentration at the main wall
  call wall_density_to_concentration(sigmadivret, Zdiv, depthdiv, Cdiv)   ! OLD impurity concentration at the divertor wall

  do s = 1,1
     main_fluxes_sput(s) = (tsuold+dsulold) * (ymain(s)*Cmain)   ! sputtered flux from main wall due to impurity self-bombardment in s^-1
     div_fluxes_sput(s) = (1.-rcmb)*(dsuold+rcls) * (ydiv(s)*Cdiv)   ! sputtered flux from divertor wall due to impurity self-bombardment in s^-1
     rclw_sput = main_fluxes_sput(s)
     rcld_sput = div_fluxes_sput(s)
  enddo
 
  do s = 2,1+species
     main_fluxes_sput(s) = fluxmain(s-1) * (ymain(s)*Cmain)   ! sputtered fluxes from main wall due to background bombardment in s^-1
     rclw_sput = rclw_sput + main_fluxes_sput(s)   ! total sputtered flux from main wall in s^-1 (updated) (NEW source for core plasma)
     div_fluxes_sput(s) = fluxdiv(s-1) * (ydiv(s)*Cdiv)   ! sputtered fluxes from main wall due to background bombardment in s^-1
     rcld_sput = rcld_sput + div_fluxes_sput(s)   ! total sputtered flux from divertor wall in s^-1 (updated) (NEW source for divertor reservoir)
  enddo

  Nmainret = Nmainret + main_flux_impl*det - rclw_sput*det   ! NEW absolute number of particles retained at the main wall
  Ndivret = Ndivret + div_flux_impl*det - rcld_sput*det   ! NEW absolute number of particles retained at the main wall

  rclw = rclw_refl + rclw_recl + rclw_sput   ! total recycled flux from main wall which will be a NEW source for the core plasma in s^-1
  rcld = rcld_refl + rcld_recl + rcld_sput   ! total recycled flux from main wall which will be a NEW source for the divertor reservoir in s^-1

  ! convert again contents into cm^-1
  Nmainret = Nmainret / circ
  Ndivret = Ndivret / circ

  ! convert again fluxes into cm^-1 s^-1
  rclw_refl = rclw_refl / circ
  rcld_refl = rcld_refl / circ
  rclw_recl = rclw_recl / circ
  rcld_recl = rcld_recl / circ
  main_flux_impl = main_flux_impl / circ
  div_flux_impl = div_flux_impl / circ
  do s = 1,1+species
     main_fluxes_sput(s) = main_fluxes_sput(s) / circ
     div_fluxes_sput(s) = div_fluxes_sput(s) / circ
  enddo
  rclw = rclw / circ
  rcld = rcld / circ
  
  return
  
end subroutine advanced_PWI_model






!============================================================================================|
subroutine wall_density_to_concentration(sigma, Z_wall, depth, C)
  ! -----------------------------------------------------------------------------------------|
  !  Calculate the concentration of the implanted impurity into a bulk material
  !  starting from a given value of surface implantation density.
  !  The conversion will of course depend on the implantation depth.
  !
  !       Input:
  !       ----------
  !       sigma :
  !           Surface implantation density of the impurity in the bulk material, in m^-2.
  !       Z_wall : 
  !           Atomic number of the wall material.
  !       depth : 
  !           Depth of a uniform implantation profile of the impurity in the material, in A.
  !           
  !       Output:
  !       -------
  !       C: 
  !           Concentration of the impurity in the bulk material,
  !           in number of impurity atoms / total number of material atoms (bulk+imp)
  !
  ! -----------------------------------------------------------------------------------------|
  implicit none
  
  ! ----- I/O variables ---------------------------------------------------------------------|
  
  real*8,  intent(in)  :: sigma
  integer, intent(in)  :: Z_wall
  real*8,  intent(in)  :: depth
  
  real*8,  intent(out) :: C 
  
  ! ----- Local variables -------------------------------------------------------------------|
  real*8     :: rho
  real*8     :: M  
  real*8     :: avogadro  
  real*8     :: mol_density
  real*8     :: n_bulk
  real*8     :: depth_m
  real*8     :: n_imp
    
  
  ! density and molar mass of the bulk material
  
  if (Z_wall == 4) then   ! beryllium
     rho = 1.80   ! g/cm^3
     M = 9.01     ! g/mol
  endif
  if (Z_wall == 6) then   ! carbon
     rho = 1.85   ! g/cm^3
     M = 12.01    ! g/mol
  endif
  if (Z_wall == 74) then   ! tungsten
     rho = 19.29  ! g/cm^3
     M = 183.85   ! g/mol
  endif
        
  avogadro = 6.022E23   ! mol^-1
    
  ! calculate the molar density
  mol_density = rho / M   ! mol/cm^3
    
  ! calculate the bulk atom density
  n_bulk = avogadro * mol_density * 1.0E6   ! atoms/m^-3
    
  ! convert the surface implantation density into volumetric impurity atom density
  depth_m = depth * 1.0E-10   ! m
  n_imp = sigma/depth_m
    
  ! calculate the resulting impurity concentration
  C = (n_imp) / (n_imp + n_bulk)
  
  return
  
end subroutine wall_density_to_concentration
