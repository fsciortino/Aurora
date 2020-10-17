 subroutine get_radial_grid( &
     ir, dbound, k, dr_0, dr_1, rvol_lcfs, &
     rr, pro, prox, qpr)
   !
   !  Subroutine to produce radial grid as in STRAHL (see plad.f)
   !  NB: output grid arrays will be padded with zeros at the end.
   !
   ! NOTE: this function isn't needed anymore, since create_radial_grid in grids_utils.py
   ! now does the same thing in Python. This subroutine will be deleted soon!
   !

   IMPLICIT NONE

   INTEGER, INTENT(INOUT)                 :: ir  ! number of grid points  -- only used if dr_0<0
   REAL*8, INTENT(IN)                     :: dbound ! from param file

   REAL*8, INTENT(IN)                     :: k
   REAL*8, INTENT(IN)                     :: dr_0 ! from param file
   REAL*8, INTENT(IN)                     :: dr_1 ! from param file
   REAL*8, INTENT(IN)                     :: rvol_lcfs

   REAL*8, INTENT(OUT)                    :: rr(1000)   ! hard-coded max number of elements
   REAL*8, INTENT(OUT)                    :: pro(1000)  ! hard-coded max number of elements
   REAL*8, INTENT(OUT)                    :: prox
   REAL*8, INTENT(OUT)                    :: qpr(1000)  ! hard-coded max number of elements

   REAL*8 :: dro, rx, temp1, temp2, temp3
   integer :: i, j
   REAL*8 :: rrx(1000), ro(1000) ! hard-coded max number of elements
   REAL*8 :: a0,a1, der

   rx=rvol_lcfs+dbound

   ! Option #1
   if (dr_0.le.0.d0) then
      dro=(rx**k)/(float(ir)-1.5)
      der=dro**(1./k)

      do i=1,ir
         ro(i)=(i-1)*dro
         rr(i)=ro(i)**(1./k)
         rrx(i)=rr(i)/rvol_lcfs
      enddo

      ! terms related with derivatives of rho
      temp1 = .5/dro
      pro(1)=2./der**2
      do i=2,ir
         pro(i)=temp1*k*rr(i)**(k-1.)
         qpr(i)=k*pro(i)/rr(i)
      enddo
      prox = k*rx**(k-1.)/dro
   endif

   ! Option #2
   if (dr_0.gt.0.d0) then
      der = dr_0
      dro = 1.d0
      a0=1./dr_0
      a1=1./dr_1
      ir=nint(1.5+rx*(a0*k+a1)/(k+1.))

      a1=(k+1.)*(float(ir)-1.5)/rx-a0*k

      ! radius
      rr(1)=0.
      do i=2,ir
         temp1=0.d0
         temp2=rx*1.05
         ro(i) = (i-1)*dro
         do j=1,50
            rr(i) = (temp1+temp2)/2.
            temp3 = a0*rr(i)+(a1-a0)*rx/(k+1.)*(rr(i)/rx)**(k+1.)
            if (temp3.ge.ro(i)) then
               temp2=rr(i)
            else
               temp1=rr(i)
            endif
         enddo
         rrx(i)=rr(i)/rvol_lcfs
      enddo

      ! terms related with derivatives of rho
      temp1=.5/dro
      pro(1)=2./der**2
      do i=2,ir
         pro(i)=(a0+(a1-a0)*(rr(i)/rx)**k)*temp1
         qpr(i)=pro(i)/rr(i)+temp1*(a1-a0)*k/rx*(rr(i)/rx)**(k-1.)
      enddo
      prox = a1/dro
   endif

   return
 end subroutine get_radial_grid






 
subroutine time_steps(n, t, dtstart, itz, tinc, verbose, t_vals, i_save)
  !
  ! Define time base for Aurora based on user inputs
  ! This function reproduces the functionality of STRAHL's time_steps.f
  ! Refer to the STRAHL manual for definitions of the time grid
  !
  ! INPUTS:
  ! n             :      number of elements in time definition arrays
  ! t(n)          :      time vector of the time base changes
  ! dtstart(n)    :      dt value at the start of a cycle
  ! itz(n)        :      cycle length, i.e. number of time steps before increasing dt
  ! tinc(n)       :      factor by which time steps should be increasing within a cycle
  !
  ! OPTIONAL:
  ! verbose       :      print to terminal a few extra info
  !
  ! OUTPUTS:
  ! t_vals          :      time base
  !
  IMPLICIT NONE

  INTEGER, INTENT(IN)            :: n
  REAL*8, INTENT(IN)             :: t(n)
  REAL*8, INTENT(IN)             :: dtstart(n)
  INTEGER, INTENT(IN)            :: itz(n)
  REAL*8, INTENT(IN)             :: tinc(n)
  LOGICAL, INTENT(IN), OPTIONAL  :: verbose

  REAL*8, INTENT(OUT)            :: t_vals(30000)   ! hard-coded max size! Seg-fault if exceeded
  INTEGER, INTENT(OUT)           :: i_save(30000)   ! hard-coded max size! Seg-fault if exceeded

  INTEGER :: nsteps
  REAL*8 :: dt(250),ncyctot(250)   ! hard-coded max number of changes to 250

  REAL*8 ::  f ,wksp_r(n), det, tnew
  INTEGER :: i, ncyc(250), nevent, wksp_i(n), index(n), m , n_itz, nn
  LOGICAL :: verb

  ! arrays wihtout double time points
  REAL*8 :: t_s(250), dtstart_s(250), tinc_s(250)
  INTEGER:: itz_s(250)

  if(present(verbose))then
     verb=verbose
  else
     verb=.false.
  endif

  ! sort t-change and related verctors
  call asc_sort(n, t, index)
  call chg_ord_r(n, dtstart, index, wksp_r) ! modifies in place
  call chg_ord_r(n, tinc, index, wksp_r)    ! modifies in place
  call chg_ord_i(n, itz, index, wksp_i)     ! modifies in place

  if (verb) then
     write(*,*) 'Inputs after sorting::'
     write(*,*) 't:',t
     write(*,*) 'dtstart: ', dtstart
     write(*,*) 'tinc: ', tinc
     write(*,*) 'itz: ', itz
  endif

  ! cancel double time points
  nevent = 1
  t_s(1) = t(1)
  dtstart_s(1) = dtstart(1)
  tinc_s(1) = tinc(1)
  itz_s(1) = itz(1)
  do i= 2, n
     if ( abs(t(i)-t(nevent)) .gt. 1.e-8) then
        nevent = nevent + 1
        t_s(nevent) = t(i)
        dtstart_s(nevent) = dtstart(i)
        tinc_s(nevent) = tinc(i)
        itz_s(nevent) = itz(i)
     endif
  enddo

  ! define # of cycles for every interval with a start time step: dtstart(i)

  do i=1, nevent-1
     f = (t_s(i+1) - t_s(i))/dtstart_s(i)
     if (i.eq.1) f = f + 1.
     if (tinc_s(i).gt.1.) then
        ncyc(i) = max(2,nint(log(f/itz_s(i)*(tinc_s(i)-1.)+1.)/log(tinc_s(i))))
     endif
     if (tinc_s(i).eq.1.) then
        ncyc(i) = max(2,nint(f/itz_s(i)))
     endif

     if (i.eq.1) ncyctot(i) = ncyc(i)
     if (i.gt.1) ncyctot(i) = ncyctot(i-1) + ncyc(i)
  enddo

  ! sum of all timesteps
  nsteps = 0
  do i= 1, nevent-1
     nsteps = nsteps + ncyc(i) * itz_s(i)
  enddo
  nsteps = nsteps - 1

  ! define real start timestep dt(i) to fit the time intervals
  do i=1, nevent-1
     if (tinc_s(i).gt.1.) then
        f = itz_s(i)* (tinc_s(i)**ncyc(i)-1.)/(tinc_s(i)-1.)
     endif
     if (tinc_s(i).eq.1.) f = 1. * itz_s(i) * ncyc(i)
     if (i.eq.1) f = f - 1.
     dt(i) = (t_s(i+1) - t_s(i))/f
  enddo

  ! Now, define t list
  nn = 1
  n_itz = 1
  m = 1
  tnew = t_s(1)
  nevent=1
  det = dt(1)

  if (verb) then
     write(*,*) 'time_steps:'
     write(*,*) 'nsteps:',nsteps
     do i=1, nevent-1
        write(*,*) 'Interval:',i,' start step: ',dt(i)
     enddo
  endif

  t_vals(1) = tnew
  i_save(1) = 1

100 tnew = tnew + det
  t_vals(nn+1) = tnew

  if (mod(n_itz+1,itz_s(nevent)) .ne. 0) then
     nn = nn+1
     n_itz = n_itz + 1
     goto 100
  endif
  n_itz = 0
  det = tinc(nevent) * det
  if (m .eq. ncyctot(nevent) .and. nn.ne.nsteps) then
     nevent = nevent + 1
     det = dt(nevent)
  endif
  m=m+1
  i_save(nn+1) = 1

  if (nn.lt.nsteps) then
     nn = nn + 1
     goto 100
  endif

  return
end subroutine time_steps
