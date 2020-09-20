
subroutine linip_arr(m, n, ts, ys, t, y)
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
  IMPLICIT NONE

  INTEGER, INTENT(IN)                 :: m
  INTEGER, INTENT(IN)                 :: n
  REAL*8, INTENT(IN)                   :: ts(m)
  REAL*8, INTENT(IN)                   :: ys(n, m)

  REAL*8, INTENT(IN)                   :: t
  REAL*8, INTENT(OUT)                :: y(n)

  INTEGER :: k, i

  REAL*8 ::  ff

  ! if only one time point has been given, then return the same/only value
  if (m.eq.1) then
     do i=1,n
        y(i) = ys(i,1)
     enddo
     return
  endif

  ! if the requested time is greater than the maximum time provided, then use the last time
  if (t.ge.ts(m)) then
     do i=1,n
        y(i) = ys(i,m)
     enddo
     return
  endif

  ! if the requested time is smaller than the minimum time provided, then use the first time
  if (t.lt.ts(1)) then
     do i=1,n
        y(i) = ys(i,1)
     enddo
     return
  endif

  ! find time point in "ts" vector that is marginally larger than the requested "t"
  k=0
  do k=1,m+1
     if (t < ts(k+1)) then
        exit
     endif
  enddo

  ff = (t - ts(k))/(ts(k+1) - ts(k))
  do i=1,n
     y(i) = ys(i,k)*(1.0d0 - ff) + ys(i,k+1)*ff
  enddo

end subroutine linip_arr









subroutine asc_sort(n, a, index)
  ! sorts a in ascending order and gives the indices
  ! of the old vector for the new vector

  IMPLICIT NONE

  INTEGER, INTENT(IN)           :: n
  REAL*8, INTENT(INOUT)         :: a(n)
  INTEGER, INTENT(OUT)          :: index(n)

  REAL*8 ::  v
  INTEGER :: i,j,w

  do i=1,n
     index(i) = i
  enddo

  do j=2, n
     v = a(j)
     w = index(j)
     do i= j-1,  1, -1
        if (a(i) .le. v) goto 10
        a(i+1) = a(i)
        index(i+1) = index(i)
     enddo
     i = 0
10   a(i+1) = v
     index(i+1) = w
  enddo
  return
end subroutine asc_sort




subroutine chg_ord_r(n, a, index, wksp)
  ! changes order of real vector a according to index vector

  IMPLICIT NONE
  INTEGER, INTENT(IN)       :: n
  REAL*8, INTENT(INOUT)     :: a(n)
  INTEGER, INTENT(IN)       :: index(n)
  REAL*8, INTENT(OUT)       :: wksp(n)

  INTEGER :: i

  do i=1, n
     wksp(i) = a(index(i))
  enddo
  do i=1, n
     a(i) = wksp(i)
  enddo
  return
end subroutine chg_ord_r



subroutine chg_ord_i(n, a, index, wksp)
  ! changes order of integer vector a according to index vector

  IMPLICIT NONE
  INTEGER, INTENT(IN)    :: n
  INTEGER, INTENT(INOUT)     :: a(n)
  INTEGER, INTENT(IN)    :: index(n)
  INTEGER, INTENT(OUT)    :: wksp(n)

  INTEGER :: i

  do i=1, n
     wksp(i) = a(index(i))
  enddo
  do i=1, n
     a(i) = wksp(i)
  enddo
  return
end subroutine chg_ord_i

