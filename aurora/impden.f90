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

! Subroutines to forward model impurity transport. 
!
! impden0 uses the standard finite-difference scheme used in STRAHL.
! impden1 uses the finite-volumes scheme described in Linder NF 2020. 
!

subroutine impden0(nion, ir, ra, rn, diff, conv, par_loss_rate, &
     src_prof, rcl_rad_prof, s_rates, r_rates,  &
     rr, pro, qpr, dlen, det,  &    ! renaming dt-->det. In this subroutine, dt is half-step
     rcl,tsuold, dsulold, divold, taudiv,tauwret, &
     a, b, c, d1, bet, gam, &
     Nret, rcld,rclw)
  !
  !  Impurity transport forward modeling with default STRAHL finite-differences scheme.
  !  Refer to STRAHL manual 2018 for details.
  !
  ! ----------------------------------------------------------------------|

  IMPLICIT NONE

  INTEGER, INTENT(IN)       :: nion
  INTEGER, INTENT(IN)       :: ir
  REAL*8, INTENT(IN)        :: ra(ir,nion)   ! alt = old
  REAL*8, INTENT(OUT)       :: rn(ir,nion)   ! neu = new
  REAL*8, INTENT(IN)        :: diff(ir, nion)
  REAL*8, INTENT(IN)        :: conv(ir, nion)
  REAL*8, INTENT(IN)        :: par_loss_rate(ir)
  REAL*8, INTENT(IN)        :: src_prof(ir)
  REAL*8, INTENT(IN)        :: rcl_rad_prof(ir)
  REAL*8, INTENT(IN)        :: s_rates(ir,nion)
  REAL*8, INTENT(IN)        :: r_rates(ir,nion)
  REAL*8, INTENT(IN)        :: rr(ir)
  REAL*8, INTENT(IN)        :: pro(ir)
  REAL*8, INTENT(IN)        :: qpr(ir)
  REAL*8, INTENT(IN)        :: dlen ! decay length at last radial grid point
  REAL*8, INTENT(IN)        :: det   ! full time step, named dt in subroutine `run'

  ! recycling parameters
  REAL*8, INTENT(IN)        :: rcl
  REAL*8, INTENT(IN)        :: tsuold   ! tsu from previous recycling step
  REAL*8, INTENT(IN)        :: dsulold  ! dsul from previous recycling step
  REAL*8, INTENT(IN)        :: divold   ! divnew from previous step (even without recycling)

  REAL*8, INTENT(IN)        :: taudiv
  REAL*8, INTENT(IN)        :: tauwret

  ! Re-use memory allocation
  REAL*8, INTENT(INOUT)     :: a(ir,nion), b(ir,nion), c(ir,nion), d1(ir), bet(ir), gam(ir)

  REAL*8, INTENT(INOUT)     :: Nret
  REAL*8, INTENT(OUT)       :: rcld
  REAL*8, INTENT(OUT)       :: rclw
  
  REAL*8 :: der
  INTEGER :: i, nz

  REAL*8 temp1, temp2, temp3, temp4, dt

  ! dr near magnetic axis
  der = rr(2) - rr(1)

  !     **************************************************
  !     ** Time CENTERED TREATMENT in radial coordinate **
  !     ** LACKNER METHOD IN Z                          **
  !     ** time step is split in two half time steps    **
  !     **************************************************

  ! ------ Recycling ------
  ! Part of the particles that hit the wall are taken to be fully-stuck.
  ! Another part is only temporarily retained at the wall.
  ! Particles FULLY STUCK at the wall will never leave, i.e. tve can only increase over time (see above).
  ! Particles that are only temporarily retained at the wall (given by the recycling fraction) come back
  ! (recycle) according to the tauwret time scale.

  if (rcl.ge.0) then    ! activated divertor return (R>=0) + recycling mode (if R>0)
     rcld = divold/taudiv
     rclw = rcl*(tsuold+dsulold)

     if (tauwret.gt.0.0d0) then
        ! include flux from particles previously retained at the wall
        Nret = Nret * (1-det/tauwret) + rclw*det     ! number of particles temporarily retained at wall
        rclw = Nret/tauwret    ! component that goes back to be a source
     endif

  else   ! no divertor return at all
        rcld = 0.d0
        rclw = 0.d0
  endif

  ! sum radial sources from external source time history and recycling
  ! NB: in STRAHL, recycling is set to have the same radial profile as the external sources!
  rn(:,1) = src_prof + rcl_rad_prof*(rclw + rcld)

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





subroutine impden1(nion, ir, ra, rn, diff, conv, par_loss_rate, src_prof, rcl_rad_prof, s_rates, r_rates,  &
     rr, fall_outsol, det,  &    ! renaming dt-->det. In this subroutine, dt is half-step
     rcl,tsuold, dsulold, divold, taudiv,tauwret, &
     evolveneut, Nret, rcld,rclw)
  !
  !  Impurity transport forward modeling with Linder's finite-volume scheme.
  !  See Linder et al. NF 2020
  !
  ! ----------------------------------------------------------------------|
  implicit none
    
  INTEGER, INTENT(IN)       :: nion
  INTEGER, INTENT(IN)       :: ir
  REAL*8, INTENT(IN)        :: ra(ir,nion)   ! old imp density
  REAL*8, INTENT(OUT)       :: rn(ir,nion)   ! new imp density
  REAL*8, INTENT(IN)        :: diff(ir, nion)
  REAL*8, INTENT(IN)        :: conv(ir, nion)
  REAL*8, INTENT(IN)        :: par_loss_rate(ir)
  REAL*8, INTENT(IN)        :: src_prof(ir)
  REAL*8, INTENT(IN)        :: rcl_rad_prof(ir)
  REAL*8, INTENT(IN)        :: s_rates(ir,nion)    ! ionization
  REAL*8, INTENT(IN)        :: r_rates(ir,nion)   ! recombination
  REAL*8, INTENT(IN)        :: rr(ir)
  REAL*8, INTENT(IN)        :: fall_outsol  !dlen ! decay length at last radial grid point
  REAL*8, INTENT(IN)        :: det   ! full time step, named dt in subroutine `run'

  ! recycling parameters
  REAL*8, INTENT(IN)        :: rcl
  REAL*8, INTENT(IN)        :: tsuold   ! tsu from previous recycling step
  REAL*8, INTENT(IN)        :: dsulold  ! dsul from previous recycling step
  REAL*8, INTENT(IN)        :: divold   ! divnew from previous step (even without recycling)

  REAL*8, INTENT(IN)        :: taudiv
  REAL*8, INTENT(IN)        :: tauwret

  ! Extras
  LOGICAL, INTENT(IN)       :: evolveneut
  
  REAL*8, INTENT(INOUT)     :: Nret
  REAL*8, INTENT(OUT)       :: rcld
  REAL*8, INTENT(OUT)       :: rclw

  REAL*8 :: flx_rcl
  INTEGER :: i, nz, ns

  REAL*8 :: dt

  ! nt is the solution of the TDMA
  real*8 ::  nt(ir), a(ir), b(ir), c(ir), d(ir)
  
  ! Lackner method: split time step in 2 half time steps
  dt = det/2.    
  
  ! Recycling 
  if (rcl.ge.0) then    ! activated divertor return (R>=0) + recycling mode (if R>0)
     rcld = divold/taudiv
     rclw = rcl*(tsuold+dsulold)
     
     if (tauwret.gt.0.0d0) then
        ! include flux from particles previously retained at the wall
        Nret = Nret * (1-det/tauwret) + rclw*det     ! number of particles temporarily retained at wall
        rclw = Nret/tauwret    ! component that goes back to be a source
     endif
     
     flx_rcl = rclw + rcld
  else   ! no divertor return at all
     rcld = 0.d0
     rclw = 0.d0
     flx_rcl = 0.d0
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
        d(i)    = d(i) + dt*(src_prof(i) + flx_rcl*rcl_rad_prof(i)) 
     enddo
     call TDMA(a, b, c, d, ir, nt)
  else
     nt = 0.d0
     ! radial profile of neutrals (given as input)
     do i=1,ir
        rn(i,1) = src_prof(i) + flx_rcl* rcl_rad_prof(i)
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
        d(i)    = d(i) - dt*nt(i)*s_rates(i,1) + dt*(src_prof(i) + flx_rcl* rcl_rad_prof(i))
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
