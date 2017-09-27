module quadrature
  use iso_fortran_env
  implicit none
  ! 15 June 2016
  ! quadrature module
  ! routines from QUADPACK, fortran90 version:
  ! https://people.sc.fsu.edu/~jburkardt/f_src/quadpack/quadpack.f90
  ! kind types parameters from the iso_fortran_env intrinsic module
  ! Author: Simone Marsili

contains

  subroutine dqagie ( f, bound, inf, epsabs, epsrel, limit, result, abserr, &
       neval, ier, alist, blist, rlist, elist, iord, last )

    !*****************************************************************************80
    !
    !! DQAGIE estimates an integral over a semi-infinite or infinite interval.
    !
    !  Modified:
    !
    !    11 September 2015
    !
    !  Author:
    !
    !    Robert Piessens, Elise de Doncker
    !
    !***purpose  the routine calculates an approximation result to a given
    !      integral   i = integral of f over (bound,+infinity)
    !      or i = integral of f over (-infinity,bound)
    !      or i = integral of f over (-infinity,+infinity),
    !      hopefully satisfying following claim for accuracy
    !      abs(i-result).le.max(epsabs,epsrel*abs(i))
    !
    !  Parameters:
    !
    !      f      - real ( kind = 8 )
    !               function subprogram defining the integrand
    !               function f(x). the actual name for f needs to be
    !               declared e x t e r n a l in the driver program.
    !
    !      bound  - real ( kind = 8 )
    !               finite bound of integration range
    !               (has no meaning if interval is doubly-infinite)
    !
    !      inf    - real ( kind = 8 )
    !               indicating the kind of integration range involved
    !               inf = 1 corresponds to  (bound,+infinity),
    !               inf = -1            to  (-infinity,bound),
    !               inf = 2             to (-infinity,+infinity).
    !
    !      epsabs - real ( kind = 8 )
    !               absolute accuracy requested
    !      epsrel - real ( kind = 8 )
    !               relative accuracy requested
    !               if  epsabs.le.0
    !               and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
    !               the routine will end with ier = 6.
    !
    !      limit  - integer ( kind = 4 )
    !               gives an upper bound on the number of subintervals
    !               in the partition of (a,b), limit.ge.1
    !
    !   on return
    !      result - real ( kind = 8 )
    !               approximation to the integral
    !
    !      abserr - real ( kind = 8 )
    !               estimate of the modulus of the absolute error,
    !               which should equal or exceed abs(i-result)
    !
    !      neval  - integer ( kind = 4 )
    !               number of integrand evaluations
    !
    !      ier    - integer ( kind = 4 )
    !               ier = 0 normal and reliable termination of the
    !                       routine. it is assumed that the requested
    !                       accuracy has been achieved.
    !             - ier.gt.0 abnormal termination of the routine. the
    !                       estimates for result and error are less
    !                       reliable. it is assumed that the requested
    !                       accuracy has not been achieved.
    !      error messages
    !               ier = 1 maximum number of subdivisions allowed
    !                       has been achieved. one can allow more
    !                       subdivisions by increasing the value of
    !                       limit (and taking the according dimension
    !                       adjustments into account). however,if
    !                       this yields no improvement it is advised
    !                       to analyze the integrand in order to
    !                       determine the integration difficulties.
    !                       if the position of a local difficulty can
    !                       be determined (e.g. singularity,
    !                       discontinuity within the interval) one
    !                       will probably gain from splitting up the
    !                       interval at this point and calling the
    !                       integrator on the subranges. if possible,
    !                       an appropriate special-purpose integrator
    !                       should be used, which is designed for
    !                       handling the type of difficulty involved.
    !                   = 2 the occurrence of roundoff error is
    !                       detected, which prevents the requested
    !                       tolerance from being achieved.
    !                       the error may be under-estimated.
    !                   = 3 extremely bad integrand behaviour occurs
    !                       at some points of the integration
    !                       interval.
    !                   = 4 the algorithm does not converge.
    !                       roundoff error is detected in the
    !                       extrapolation table.
    !                       it is assumed that the requested tolerance
    !                       cannot be achieved, and that the returned
    !                       result is the best which can be obtained.
    !                   = 5 the integral is probably divergent, or
    !                       slowly convergent. it must be noted that
    !                       divergence can occur with any other value
    !                       of ier.
    !                   = 6 the input is invalid, because
    !                       (epsabs.le.0 and
    !                        epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
    !                       result, abserr, neval, last, rlist(1),
    !                       elist(1) and iord(1) are set to zero.
    !                       alist(1) and blist(1) are set to 0
    !                       and 1 respectively.
    !
    !      alist  - real ( kind = 8 )
    !               vector of dimension at least limit, the first
    !                last  elements of which are the left
    !               end points of the subintervals in the partition
    !               of the transformed integration range (0,1).
    !
    !      blist  - real ( kind = 8 )
    !               vector of dimension at least limit, the first
    !                last  elements of which are the right
    !               end points of the subintervals in the partition
    !               of the transformed integration range (0,1).
    !
    !      rlist  - real ( kind = 8 )
    !               vector of dimension at least limit, the first
    !                last  elements of which are the integral
    !               approximations on the subintervals
    !
    !      elist  - real ( kind = 8 )
    !               vector of dimension at least limit,  the first
    !               last elements of which are the moduli of the
    !               absolute error estimates on the subintervals
    !
    !      iord   - integer ( kind = 4 )
    !               vector of dimension limit, the first k
    !               elements of which are pointers to the
    !               error estimates over the subintervals,
    !               such that elist(iord(1)), ..., elist(iord(k))
    !               form a decreasing sequence, with k = last
    !               if last.le.(limit/2+2), and k = limit+1-last
    !               otherwise
    !
    !      last   - integer ( kind = 4 )
    !               number of subintervals actually produced
    !               in the subdivision process
    !
    !  Local Parameters:
    !
    !      the dimension of rlist2 is determined by the value of
    !      limexp in routine dqelg.
    !
    !     alist     - list of left end points of all subintervals
    !                 considered up to now
    !     blist     - list of right end points of all subintervals
    !                 considered up to now
    !     rlist(i)  - approximation to the integral over
    !                 (alist(i),blist(i))
    !     rlist2    - array of dimension at least (limexp+2),
    !                 containing the part of the epsilon table
    !                 wich is still needed for further computations
    !     elist(i)  - error estimate applying to rlist(i)
    !     maxerr    - pointer to the interval with largest error
    !                 estimate
    !     errmax    - elist(maxerr)
    !     erlast    - error on the interval currently subdivided
    !                 (before that subdivision has taken place)
    !     area      - sum of the integrals over the subintervals
    !     errsum    - sum of the errors over the subintervals
    !     errbnd    - requested accuracy max(epsabs,epsrel*
    !                 abs(result))
    !     *****1    - variable for the left subinterval
    !     *****2    - variable for the right subinterval
    !     last      - index for subdivision
    !     nres      - number of calls to the extrapolation routine
    !     numrl2    - number of elements currently in rlist2. if an
    !                 appropriate approximation to the compounded
    !                 integral has been obtained, it is put in
    !                 rlist2(numrl2) after numrl2 has been increased
    !                 by one.
    !     small     - length of the smallest interval considered up
    !                 to now, multiplied by 1.5
    !     erlarg    - sum of the errors over the intervals larger
    !                 than the smallest interval considered up to now
    !     extrap    - logical variable denoting that the routine
    !                 is attempting to perform extrapolation. i.e.
    !                 before subdividing the smallest interval we
    !                 try to decrease the value of erlarg.
    !     noext     - logical variable denoting that extrapolation
    !                 is no longer allowed (true-value)
    !
    !      machine dependent constants
    !
    !     epmach is the largest relative spacing.
    !     uflow is the smallest positive magnitude.
    !     oflow is the largest positive magnitude.
    !
    implicit none

    real (real64) abseps,abserr,alist,area,area1,area12,area2,a1, &
         a2,blist,boun,bound,b1,b2,correc,defabs,defab1,defab2, &
         dres,elist,epmach,epsabs,epsrel,erlarg,erlast, &
         errbnd,errmax,error1,error2,erro12,errsum,ertest,f,oflow,resabs, &
         reseps,result,res3la,rlist,rlist2,small,uflow
    integer (int32) id,ier,ierro,inf,iord,iroff1,iroff2, &
         iroff3,jupbnd,k,ksgn, &
         ktmin,last,limit,maxerr,neval,nres,nrmax,numrl2
    logical extrap,noext
    dimension alist(limit),blist(limit),elist(limit),iord(limit), &
         res3la(3),rlist(limit),rlist2(52)

    external f

    epmach = epsilon ( epmach )
    !
    !  test on validity of parameters
    !
    ier = 0
    neval = 0
    last = 0
    result = 0.0_real64
    abserr = 0.0_real64
    alist(1) = 0.0_real64
    blist(1) = 0.1e1_real64
    rlist(1) = 0.0_real64
    elist(1) = 0.0_real64
    iord(1) = 0

    if(epsabs.le.0.0_real64.and.epsrel.lt. max ( 0.5e2_real64*epmach,0.5D-28)) then
       ier = 6
    end if

    if(ier.eq.6) then
       return
    end if
    !
    !  first approximation to the integral
    !
    !  determine the interval to be mapped onto (0,1).
    !  if inf = 2 the integral is computed as i = i1+i2, where
    !  i1 = integral of f over (-infinity,0),
    !  i2 = integral of f over (0,+infinity).
    !
    boun = bound
    if(inf.eq.2) boun = 0.0_real64
    call dqk15i(f,boun,inf,0.0_real64,0.1e1_real64,result,abserr, &
         defabs,resabs)
    !
    !  test on accuracy
    !
    last = 1
    rlist(1) = result
    elist(1) = abserr
    iord(1) = 1
    dres =  abs ( result)
    errbnd =  max ( epsabs,epsrel*dres)
    if(abserr.le.1.0e2_real64*epmach*defabs.and.abserr.gt.errbnd) ier = 2
    if(limit.eq.1) ier = 1
    if(ier.ne.0.or.(abserr.le.errbnd.and.abserr.ne.resabs).or. &
         abserr.eq.0.0_real64) go to 130
    !
    !  initialization
    !
    uflow = tiny ( uflow )
    oflow = huge ( oflow )
    rlist2(1) = result
    errmax = abserr
    maxerr = 1
    area = result
    errsum = abserr
    abserr = oflow
    nrmax = 1
    nres = 0
    ktmin = 0
    numrl2 = 2
    extrap = .false.
    noext = .false.
    ierro = 0
    iroff1 = 0
    iroff2 = 0
    iroff3 = 0
    ksgn = -1
    if(dres.ge.(0.1e1_real64-0.5e2_real64*epmach)*defabs) ksgn = 1
    !
    !  main do-loop
    !
    do 90 last = 2,limit
       !
       !  bisect the subinterval with nrmax-th largest error estimate.
       !
       a1 = alist(maxerr)
       b1 = 0.5_real64*(alist(maxerr)+blist(maxerr))
       a2 = b1
       b2 = blist(maxerr)
       erlast = errmax
       call dqk15i(f,boun,inf,a1,b1,area1,error1,resabs,defab1)
       call dqk15i(f,boun,inf,a2,b2,area2,error2,resabs,defab2)
       !
       !  improve previous approximations to integral
       !  and error and test for accuracy.
       !
       area12 = area1+area2
       erro12 = error1+error2
       errsum = errsum+erro12-errmax
       area = area+area12-rlist(maxerr)
       if(defab1.eq.error1.or.defab2.eq.error2)go to 15
       if( abs ( rlist(maxerr)-area12).gt.0.1e-4_real64* abs ( area12) &
            .or.erro12.lt.0.99_real64*errmax) go to 10
       if(extrap) iroff2 = iroff2+1
       if(.not.extrap) iroff1 = iroff1+1
10     if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
15     rlist(maxerr) = area1
       rlist(last) = area2
       errbnd =  max ( epsabs,epsrel* abs ( area))
       !
       !  test for roundoff error and eventually set error flag.
       !
       if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
       if(iroff2.ge.5) ierro = 3
       !
       !  set error flag in the case that the number of
       !  subintervals equals limit.
       !
       if(last.eq.limit) ier = 1
       !
       !  set error flag in the case of bad integrand behaviour
       !  at some points of the integration range.
       !
       if( max (  abs ( a1), abs ( b2)).le.(0.1e1_real64+0.1e3_real64*epmach)* &
            ( abs ( a2)+0.1e4_real64*uflow)) then
          ier = 4
       end if
       !
       !  append the newly-created intervals to the list.
       !
       if(error2.gt.error1) go to 20
       alist(last) = a2
       blist(maxerr) = b1
       blist(last) = b2
       elist(maxerr) = error1
       elist(last) = error2
       go to 30
20     continue

       alist(maxerr) = a2
       alist(last) = a1
       blist(last) = b1
       rlist(maxerr) = area2
       rlist(last) = area1
       elist(maxerr) = error2
       elist(last) = error1
       !
       !  call dqpsrt to maintain the descending ordering
       !  in the list of error estimates and select the subinterval
       !  with nrmax-th largest error estimate (to be bisected next).
       !
30     call dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
       if(errsum.le.errbnd) go to 115
       if(ier.ne.0) go to 100
       if(last.eq.2) go to 80
       if(noext) go to 90
       erlarg = erlarg-erlast
       if( abs ( b1-a1).gt.small) erlarg = erlarg+erro12
       if(extrap) go to 40
       !
       !  test whether the interval to be bisected next is the
       !  smallest interval.
       !
       if( abs ( blist(maxerr)-alist(maxerr)).gt.small) go to 90
       extrap = .true.
       nrmax = 2
40     if(ierro.eq.3.or.erlarg.le.ertest) go to 60
       !
       !  the smallest interval has the largest error.
       !  before bisecting decrease the sum of the errors over the
       !  larger intervals (erlarg) and perform extrapolation.
       !
       id = nrmax
       jupbnd = last
       if(last.gt.(2+limit/2)) jupbnd = limit+3-last

       do k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
          if( abs ( blist(maxerr)-alist(maxerr)).gt.small) go to 90
          nrmax = nrmax+1
       end do
       !
       !  perform extrapolation.
       !
60     numrl2 = numrl2+1
       rlist2(numrl2) = area
       call dqelg(numrl2,rlist2,reseps,abseps,res3la,nres)
       ktmin = ktmin+1
       if(ktmin.gt.5.and.abserr.lt.0.1e-2_real64*errsum) ier = 5
       if(abseps.ge.abserr) go to 70
       ktmin = 0
       abserr = abseps
       result = reseps
       correc = erlarg
       ertest =  max ( epsabs,epsrel* abs ( reseps))
       if(abserr.le.ertest) go to 100
       !
       !  prepare bisection of the smallest interval.
       !
70     if(numrl2.eq.1) noext = .true.
       if(ier.eq.5) go to 100
       maxerr = iord(1)
       errmax = elist(maxerr)
       nrmax = 1
       extrap = .false.
       small = small*0.5_real64
       erlarg = errsum
       go to 90
80     small = 0.375_real64
       erlarg = errsum
       ertest = errbnd
       rlist2(2) = area
90     continue
       !
       !  set final result and error estimate.
       !
100    if(abserr.eq.oflow) go to 115
       if((ier+ierro).eq.0) go to 110
       if(ierro.eq.3) abserr = abserr+correc
       if(ier.eq.0) ier = 3
       if(result.ne.0.0_real64.and.area.ne.0.0_real64)go to 105
       if(abserr.gt.errsum)go to 115
       if(area.eq.0.0_real64) go to 130
       go to 110
105    if(abserr/ abs ( result).gt.errsum/ abs ( area))go to 115
       !
       !  test on divergence
       !
110    continue

       if ( ksgn .eq. (-1) .and. &
            max ( abs ( result), abs ( area)) .le. defabs*0.1e-1_real64 ) then
          go to 130
       end if

       if ( 0.1e-1_real64 .gt. (result/area) .or. &
            (result/area) .gt. 0.1e3_real64 .or. &
            errsum .gt. abs ( area) ) then
          ier = 6
       end if

       go to 130
       !
       !  compute global integral sum.
       !
115    result = 0.0_real64
       do k = 1,last
          result = result+rlist(k)
       end do
       abserr = errsum
130    continue

       neval = 30*last-15
       if(inf.eq.2) neval = 2*neval
       if(ier.gt.2) ier=ier-1

       return
     end subroutine dqagie
     subroutine dqagi ( f, bound, inf, epsabs, epsrel, result, abserr, neval, &
          ier,limit,lenw,last,iwork,work)

       !*****************************************************************************80
       !
       !! DQAGI estimates an integral over a semi-infinite or infinite interval.
       !
       !  Modified:
       !
       !    11 September 2015
       !
       !  Author:
       !
       !    Robert Piessens, Elise de Doncker
       !
       !***purpose  the routine calculates an approximation result to a given
       !      integral   i = integral of f over (bound,+infinity)
       !      or i = integral of f over (-infinity,bound)
       !      or i = integral of f over (-infinity,+infinity)
       !      hopefully satisfying following claim for accuracy
       !      abs(i-result).le.max(epsabs,epsrel*abs(i)).
       !
       !  Parameters:
       !
       !   on entry
       !      f      - real ( kind = 8 )
       !               function subprogram defining the integrand
       !               function f(x). the actual name for f needs to be
       !               declared e x t e r n a l in the driver program.
       !
       !      bound  - real ( kind = 8 )
       !               finite bound of integration range
       !               (has no meaning if interval is doubly-infinite)
       !
       !      inf    - integer ( kind = 4 )
       !               indicating the kind of integration range involved
       !               inf = 1 corresponds to  (bound,+infinity),
       !               inf = -1            to  (-infinity,bound),
       !               inf = 2             to (-infinity,+infinity).
       !
       !      epsabs - real ( kind = 8 )
       !               absolute accuracy requested
       !      epsrel - real ( kind = 8 )
       !               relative accuracy requested
       !               if  epsabs.le.0
       !               and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
       !               the routine will end with ier = 6.
       !
       !
       !   on return
       !      result - real ( kind = 8 )
       !               approximation to the integral
       !
       !      abserr - real ( kind = 8 )
       !               estimate of the modulus of the absolute error,
       !               which should equal or exceed abs(i-result)
       !
       !      neval  - integer ( kind = 4 )
       !               number of integrand evaluations
       !
       !      ier    - integer ( kind = 4 )
       !               ier = 0 normal and reliable termination of the
       !                       routine. it is assumed that the requested
       !                       accuracy has been achieved.
       !             - ier.gt.0 abnormal termination of the routine. the
       !                       estimates for result and error are less
       !                       reliable. it is assumed that the requested
       !                       accuracy has not been achieved.
       !      error messages
       !               ier = 1 maximum number of subdivisions allowed
       !                       has been achieved. one can allow more
       !                       subdivisions by increasing the value of
       !                       limit (and taking the according dimension
       !                       adjustments into account). however, if
       !                       this yields no improvement it is advised
       !                       to analyze the integrand in order to
       !                       determine the integration difficulties. if
       !                       the position of a local difficulty can be
       !                       determined (e.g. singularity,
       !                       discontinuity within the interval) one
       !                       will probably gain from splitting up the
       !                       interval at this point and calling the
       !                       integrator on the subranges. if possible,
       !                       an appropriate special-purpose integrator
       !                       should be used, which is designed for
       !                       handling the type of difficulty involved.
       !                   = 2 the occurrence of roundoff error is
       !                       detected, which prevents the requested
       !                       tolerance from being achieved.
       !                       the error may be under-estimated.
       !                   = 3 extremely bad integrand behaviour occurs
       !                       at some points of the integration
       !                       interval.
       !                   = 4 the algorithm does not converge.
       !                       roundoff error is detected in the
       !                       extrapolation table.
       !                       it is assumed that the requested tolerance
       !                       cannot be achieved, and that the returned
       !                       result is the best which can be obtained.
       !                   = 5 the integral is probably divergent, or
       !                       slowly convergent. it must be noted that
       !                       divergence can occur with any other value
       !                       of ier.
       !                   = 6 the input is invalid, because
       !                       (epsabs.le.0 and
       !                        epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
       !                        or limit.lt.1 or leniw.lt.limit*4.
       !                       result, abserr, neval, last are set to
       !                       zero. exept when limit or leniw is
       !                       invalid, iwork(1), work(limit*2+1) and
       !                       work(limit*3+1) are set to zero, work(1)
       !                       is set to a and work(limit+1) to b.
       !
       !   dimensioning parameters
       !      limit - integer ( kind = 4 )
       !              dimensioning parameter for iwork
       !              limit determines the maximum number of subintervals
       !              in the partition of the given integration interval
       !              (a,b), limit.ge.1.
       !              if limit.lt.1, the routine will end with ier = 6.
       !
       !      lenw  - integer ( kind = 4 )
       !              dimensioning parameter for work
       !              lenw must be at least limit*4.
       !              if lenw.lt.limit*4, the routine will end
       !              with ier = 6.
       !
       !      last  - integer ( kind = 4 )
       !              on return, last equals the number of subintervals
       !              produced in the subdivision process, which
       !              determines the number of significant elements
       !              actually in the work arrays.
       !
       !   work arrays
       !      iwork - integer ( kind = 4 )
       !              vector of dimension at least limit, the first
       !              k elements of which contain pointers
       !              to the error estimates over the subintervals,
       !              such that work(limit*3+iwork(1)),... ,
       !              work(limit*3+iwork(k)) form a decreasing
       !              sequence, with k = last if last.le.(limit/2+2), and
       !              k = limit+1-last otherwise
       !
       !      work  - real ( kind = 8 )
       !              vector of dimension at least lenw
       !              on return
       !              work(1), ..., work(last) contain the left
       !               end points of the subintervals in the
       !               partition of (a,b),
       !              work(limit+1), ..., work(limit+last) contain
       !               the right end points,
       !              work(limit*2+1), ...,work(limit*2+last) contain the
       !               integral approximations over the subintervals,
       !              work(limit*3+1), ..., work(limit*3)
       !               contain the error estimates.
       !
       implicit none

       real ( kind = 8 ) abserr,bound,epsabs,epsrel,f,result,work
       integer ( kind = 4 ) ier,inf,iwork,last,lenw,limit,lvl,l1,l2,l3,neval

       dimension iwork(limit),work(lenw)

       external f
       !
       !  check validity of limit and lenw.
       !
       ier = 6
       neval = 0
       last = 0
       result = 0.0_real64
       abserr = 0.0_real64
       if(limit.lt.1.or.lenw.lt.limit*4) go to 10
       !
       !  prepare call for dqagie.
       !
       l1 = limit+1
       l2 = limit+l1
       l3 = limit+l2

       call dqagie(f,bound,inf,epsabs,epsrel,limit,result,abserr, &
            neval,ier,work(1),work(l1),work(l2),work(l3),iwork,last)
       !
       !  call error handler if necessary.
       !
       lvl = 0
10     if(ier.eq.6) lvl = 1

       if(ier.ne.0) then
          call xerror('abnormal return from dqagi',26,ier,lvl)
       end if

       return
     end subroutine dqagi

     subroutine dqage ( f, a, b, epsabs, epsrel, key, limit, result, abserr, &
          neval, ier, alist, blist, rlist, elist, iord, last )

       !*****************************************************************************80
       !
       !! DQAGE estimates a definite integral.
       !
       !  Modified:
       !
       !    11 September 2015
       !
       !  Author:
       !
       !    Robert Piessens, Elise de Doncker
       !
       !***purpose  the routine calculates an approximation result to a given
       !      definite integral   i = integral of f over (a,b),
       !      hopefully satisfying following claim for accuracy
       !      abs(i-reslt).le.max(epsabs,epsrel*abs(i)).
       !
       !  Parameters:
       !
       !   on entry
       !      f      - real(real64)
       !               function subprogram defining the integrand
       !               function f(x). the actual name for f needs to be
       !               declared e x t e r n a l in the driver program.
       !
       !      a      - real(real64)
       !               lower limit of integration
       !
       !      b      - real(real64)
       !               upper limit of integration
       !
       !      epsabs - real(real64)
       !               absolute accuracy requested
       !      epsrel - real(real64)
       !               relative accuracy requested
       !               if  epsabs.le.0
       !               and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
       !               the routine will end with ier = 6.
       !
       !      key    - integer(int32)
       !               key for choice of local integration rule
       !               a gauss-kronrod pair is used with
       !                    7 - 15 points if key.lt.2,
       !                   10 - 21 points if key = 2,
       !                   15 - 31 points if key = 3,
       !                   20 - 41 points if key = 4,
       !                   25 - 51 points if key = 5,
       !                   30 - 61 points if key.gt.5.
       !
       !      limit  - integer(int32)
       !               gives an upperbound on the number of subintervals
       !               in the partition of (a,b), limit.ge.1.
       !
       !   on return
       !      result - real(real64)
       !               approximation to the integral
       !
       !      abserr - real(real64)
       !               estimate of the modulus of the absolute error,
       !               which should equal or exceed abs(i-result)
       !
       !      neval  - integer(int32)
       !               number of integrand evaluations
       !
       !      ier    - integer(int32)
       !               ier = 0 normal and reliable termination of the
       !                       routine. it is assumed that the requested
       !                       accuracy has been achieved.
       !               ier.gt.0 abnormal termination of the routine
       !                       the estimates for result and error are
       !                       less reliable. it is assumed that the
       !                       requested accuracy has not been achieved.
       !      error messages
       !               ier = 1 maximum number of subdivisions allowed
       !                       has been achieved. one can allow more
       !                       subdivisions by increasing the value
       !                       of limit.
       !                       however, if this yields no improvement it
       !                       is rather advised to analyze the integrand
       !                       in order to determine the integration
       !                       difficulties. if the position of a local
       !                       difficulty can be determined(e.g.
       !                       singularity, discontinuity within the
       !                       interval) one will probably gain from
       !                       splitting up the interval at this point
       !                       and calling the integrator on the
       !                       subranges. if possible, an appropriate
       !                       special-purpose integrator should be used
       !                       which is designed for handling the type of
       !                       difficulty involved.
       !                   = 2 the occurrence of roundoff error is
       !                       detected, which prevents the requested
       !                       tolerance from being achieved.
       !                   = 3 extremely bad integrand behaviour occurs
       !                       at some points of the integration
       !                       interval.
       !                   = 6 the input is invalid, because
       !                       (epsabs.le.0 and
       !                        epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
       !                       result, abserr, neval, last, rlist(1) ,
       !                       elist(1) and iord(1) are set to zero.
       !                       alist(1) and blist(1) are set to a and b
       !                       respectively.
       !
       !      alist   - real(real64)
       !                vector of dimension at least limit, the first
       !                 last  elements of which are the left
       !                end points of the subintervals in the partition
       !                of the given integration range (a,b)
       !
       !      blist   - real(real64)
       !                vector of dimension at least limit, the first
       !                 last  elements of which are the right
       !                end points of the subintervals in the partition
       !                of the given integration range (a,b)
       !
       !      rlist   - real(real64)
       !                vector of dimension at least limit, the first
       !                 last  elements of which are the
       !                integral approximations on the subintervals
       !
       !      elist   - real(real64)
       !                vector of dimension at least limit, the first
       !                 last  elements of which are the moduli of the
       !                absolute error estimates on the subintervals
       !
       !      iord    - integer(int32)
       !                vector of dimension at least limit, the first k
       !                elements of which are pointers to the
       !                error estimates over the subintervals,
       !                such that elist(iord(1)), ...,
       !                elist(iord(k)) form a decreasing sequence,
       !                with k = last if last.le.(limit/2+2), and
       !                k = limit+1-last otherwise
       !
       !      last    - integer(int32)
       !                number of subintervals actually produced in the
       !                subdivision process
       !
       !  Local Parameters:
       !
       !     alist     - list of left end points of all subintervals
       !                 considered up to now
       !     blist     - list of right end points of all subintervals
       !                 considered up to now
       !     rlist(i)  - approximation to the integral over
       !                (alist(i),blist(i))
       !     elist(i)  - error estimate applying to rlist(i)
       !     maxerr    - pointer to the interval with largest
       !                 error estimate
       !     errmax    - elist(maxerr)
       !     area      - sum of the integrals over the subintervals
       !     errsum    - sum of the errors over the subintervals
       !     errbnd    - requested accuracy max(epsabs,epsrel*
       !                 abs(result))
       !     *****1    - variable for the left subinterval
       !     *****2    - variable for the right subinterval
       !     last      - index for subdivision
       !
       !
       !     machine dependent constants
       !
       !     epmach  is the largest relative spacing.
       !     uflow  is the smallest positive magnitude.
       !
       implicit none

       real(real64) a,abserr,alist,area,area1,area12,area2,a1,a2,b, &
            blist,b1,b2,defabs,defab1,defab2,elist,epmach, &
            epsabs,epsrel,errbnd,errmax,error1,error2,erro12,errsum, &
            resabs,result,rlist,uflow
       integer(int32) ier,iord,iroff1,iroff2,k,key,keyf,last,limit, &
            maxerr, nrmax, neval
       real(real64), external :: f

       !    interface
       !       function f(x)
       !         use iso_fortran_env
       !         real(real64), intent(in) :: x
       !         real(real64) :: f
       !       end function f
       !    end interface

       dimension alist(limit),blist(limit),elist(limit),iord(limit), &
            rlist(limit)

       epmach = epsilon ( epmach )
       uflow = tiny ( uflow )
       !
       !  test on validity of parameters
       !
       ier = 0
       neval = 0
       last = 0
       result = 0.0_real64
       abserr = 0.0_real64
       alist(1) = a
       blist(1) = b
       rlist(1) = 0.0_real64
       elist(1) = 0.0_real64
       iord(1) = 0

       if(epsabs.le.0.0_real64.and. &
            epsrel.lt. max ( 0.5e2_real64*epmach,0.5e-28_real64)) then
          ier = 6
          return
       end if
       !
       !  first approximation to the integral
       !
       keyf = key
       if(key.le.0) keyf = 1
       if(key.ge.7) keyf = 6
       neval = 0
       if(keyf.eq.1) call dqk15(f,a,b,result,abserr,defabs,resabs)
       if(keyf.eq.2) call dqk21(f,a,b,result,abserr,defabs,resabs)
       if(keyf.eq.3) call dqk31(f,a,b,result,abserr,defabs,resabs)
       if(keyf.eq.4) call dqk41(f,a,b,result,abserr,defabs,resabs)
       if(keyf.eq.5) call dqk51(f,a,b,result,abserr,defabs,resabs)
       if(keyf.eq.6) call dqk61(f,a,b,result,abserr,defabs,resabs)
       last = 1
       rlist(1) = result
       elist(1) = abserr
       iord(1) = 1
       !
       !  test on accuracy.
       !
       errbnd =  max ( epsabs, epsrel* abs ( result ) )

       if(abserr.le.0.5e2_real64 * epmach * defabs .and. &
            abserr.gt.errbnd) then
          ier = 2
       end if

       if(limit.eq.1) then
          ier = 1
       end if

       if ( ier .ne. 0 .or. &
            (abserr .le. errbnd .and. abserr .ne. resabs ) .or. &
            abserr .eq. 0.0_real64 ) then

          if(keyf.ne.1) then
             neval = (10*keyf+1)*(2*neval+1)
          else
             neval = 30*neval+15
          end if

          return

       end if
       !
       !  initialization
       !
       errmax = abserr
       maxerr = 1
       area = result
       errsum = abserr
       nrmax = 1
       iroff1 = 0
       iroff2 = 0
       !
       !  main do-loop
       !
       do last = 2, limit
          !
          !  bisect the subinterval with the largest error estimate.
          !
          a1 = alist(maxerr)
          b1 = 0.5_real64*(alist(maxerr)+blist(maxerr))
          a2 = b1
          b2 = blist(maxerr)

          if(keyf.eq.1) call dqk15(f,a1,b1,area1,error1,resabs,defab1)
          if(keyf.eq.2) call dqk21(f,a1,b1,area1,error1,resabs,defab1)
          if(keyf.eq.3) call dqk31(f,a1,b1,area1,error1,resabs,defab1)
          if(keyf.eq.4) call dqk41(f,a1,b1,area1,error1,resabs,defab1)
          if(keyf.eq.5) call dqk51(f,a1,b1,area1,error1,resabs,defab1)
          if(keyf.eq.6) call dqk61(f,a1,b1,area1,error1,resabs,defab1)

          if(keyf.eq.1) call dqk15(f,a2,b2,area2,error2,resabs,defab2)
          if(keyf.eq.2) call dqk21(f,a2,b2,area2,error2,resabs,defab2)
          if(keyf.eq.3) call dqk31(f,a2,b2,area2,error2,resabs,defab2)
          if(keyf.eq.4) call dqk41(f,a2,b2,area2,error2,resabs,defab2)
          if(keyf.eq.5) call dqk51(f,a2,b2,area2,error2,resabs,defab2)
          if(keyf.eq.6) call dqk61(f,a2,b2,area2,error2,resabs,defab2)
          !
          !  improve previous approximations to integral
          !  and error and test for accuracy.
          !
          neval = neval+1
          area12 = area1+area2
          erro12 = error1+error2
          errsum = errsum+erro12-errmax
          area = area+area12-rlist(maxerr)

          if ( defab1 .ne. error1 .and. defab2 .ne. error2 ) then

             if( abs ( rlist(maxerr)-area12).le.0.1e-4_real64* abs ( area12) &
                  .and. erro12.ge.0.99_real64*errmax) then
                iroff1 = iroff1+1
             end if

             if(last.gt.10.and.erro12.gt.errmax) then
                iroff2 = iroff2+1
             end if

          end if

          rlist(maxerr) = area1
          rlist(last) = area2
          errbnd =  max ( epsabs,epsrel* abs ( area))

          if ( errbnd .lt. errsum ) then
             !
             !  test for roundoff error and eventually set error flag.
             !
             if(iroff1.ge.6.or.iroff2.ge.20) then
                ier = 2
             end if
             !
             !  set error flag in the case that the number of subintervals
             !  equals limit.
             !
             if(last.eq.limit) then
                ier = 1
             end if
             !
             !  set error flag in the case of bad integrand behaviour
             !  at a point of the integration range.
             !
             if( max (  abs ( a1), abs ( b2)).le.(0.1e1_real64+0.1e3_real64* &
                  epmach)*( abs ( a2)+0.1e4_real64*uflow)) then
                ier = 3
             end if

          end if
          !
          !  append the newly-created intervals to the list.
          !
          if(error2.le.error1) then
             alist(last) = a2
             blist(maxerr) = b1
             blist(last) = b2
             elist(maxerr) = error1
             elist(last) = error2
          else
             alist(maxerr) = a2
             alist(last) = a1
             blist(last) = b1
             rlist(maxerr) = area2
             rlist(last) = area1
             elist(maxerr) = error2
             elist(last) = error1
          end if
          !
          !  call dqpsrt to maintain the descending ordering
          !  in the list of error estimates and select the subinterval
          !  with the largest error estimate (to be bisected next).
          !
          call dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)

          if(ier.ne.0.or.errsum.le.errbnd) then
             exit
          end if

       end do
       !
       !  compute final result.
       !
       result = 0.0_real64
       do k=1,last
          result = result+rlist(k)
       end do
       abserr = errsum

       if(keyf.ne.1) then
          neval = (10*keyf+1)*(2*neval+1)
       else
          neval = 30*neval+15
       end if

       return
     end subroutine dqage
     subroutine dqag ( f, a, b, epsabs, epsrel, key, result, abserr, neval, ier, &
          limit, lenw, last, iwork, work )

       !*****************************************************************************80
       !
       !! DQAG approximates an integral over a finite interval.
       !
       !  Modified:
       !
       !    11 September 2015
       !
       !  Author:
       !
       !    Robert Piessens, Elise de Doncker
       !
       !***purpose  the routine calculates an approximation result to a given
       !      definite integral i = integral of f over (a,b),
       !      hopefully satisfying following claim for accuracy
       !      abs(i-result)le.max(epsabs,epsrel*abs(i)).
       !
       !  Parameters:
       !
       !      f      - real(real64)
       !               function subprogam defining the integrand
       !               function f(x). the actual name for f needs to be
       !               declared e x t e r n a l in the driver program.
       !
       !      a      - real(real64)
       !               lower limit of integration
       !
       !      b      - real(real64)
       !               upper limit of integration
       !
       !      epsabs - real(real64)
       !               absolute accoracy requested
       !      epsrel - real(real64)
       !               relative accuracy requested
       !               if  epsabs.le.0
       !               and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
       !               the routine will end with ier = 6.
       !
       !      key    - integer(int32)
       !               key for choice of local integration rule
       !               a gauss-kronrod pair is used with
       !                 7 - 15 points if key.lt.2,
       !                10 - 21 points if key = 2,
       !                15 - 31 points if key = 3,
       !                20 - 41 points if key = 4,
       !                25 - 51 points if key = 5,
       !                30 - 61 points if key.gt.5.
       !
       !   on return
       !      result - real(real64)
       !               approximation to the integral
       !
       !      abserr - real(real64)
       !               estimate of the modulus of the absolute error,
       !               which should equal or exceed abs(i-result)
       !
       !      neval  - integer(int32)
       !               number of integrand evaluations
       !
       !      ier    - integer(int32)
       !               ier = 0 normal and reliable termination of the
       !                       routine. it is assumed that the requested
       !                       accuracy has been achieved.
       !               ier.gt.0 abnormal termination of the routine
       !                       the estimates for result and error are
       !                       less reliable. it is assumed that the
       !                       requested accuracy has not been achieved.
       !                error messages
       !               ier = 1 maximum number of subdivisions allowed
       !                       has been achieved. one can allow more
       !                       subdivisions by increasing the value of
       !                       limit (and taking the according dimension
       !                       adjustments into account). however, if
       !                       this yield no improvement it is advised
       !                       to analyze the integrand in order to
       !                       determine the integration difficulaties.
       !                       if the position of a local difficulty can
       !                       be determined (i.e.singularity,
       !                       discontinuity within the interval) one
       !                       will probably gain from splitting up the
       !                       interval at this point and calling the
       !                       integrator on the subranges. if possible,
       !                       an appropriate special-purpose integrator
       !                       should be used which is designed for
       !                       handling the type of difficulty involved.
       !                   = 2 the occurrence of roundoff error is
       !                       detected, which prevents the requested
       !                       tolerance from being achieved.
       !                   = 3 extremely bad integrand behaviour occurs
       !                       at some points of the integration
       !                       interval.
       !                   = 6 the input is invalid, because
       !                       (epsabs.le.0 and
       !                        epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
       !                       or limit.lt.1 or lenw.lt.limit*4.
       !                       result, abserr, neval, last are set
       !                       to zero.
       !                       except when lenw is invalid, iwork(1),
       !                       work(limit*2+1) and work(limit*3+1) are
       !                       set to zero, work(1) is set to a and
       !                       work(limit+1) to b.
       !
       !   dimensioning parameters
       !      limit - integer(int32)
       !              dimensioning parameter for iwork
       !              limit determines the maximum number of subintervals
       !              in the partition of the given integration interval
       !              (a,b), limit.ge.1.
       !              if limit.lt.1, the routine will end with ier = 6.
       !
       !      lenw  - integer(int32)
       !              dimensioning parameter for work
       !              lenw must be at least limit*4.
       !              if lenw.lt.limit*4, the routine will end with
       !              ier = 6.
       !
       !      last  - integer(int32)
       !              on return, last equals the number of subintervals
       !              produced in the subdivision process, which
       !              determines the number of significant elements
       !              actually in the work arrays.
       !
       !   work arrays
       !      iwork - integer(int32)
       !              vector of dimension at least limit, the first k
       !              elements of which contain pointers to the error
       !              estimates over the subintervals, such that
       !              work(limit*3+iwork(1)),... , work(limit*3+iwork(k))
       !              form a decreasing sequence with k = last if
       !              last.le.(limit/2+2), and k = limit+1-last otherwise
       !
       !      work  - real(real64)
       !              vector of dimension at least lenw
       !              on return
       !              work(1), ..., work(last) contain the left end
       !              points of the subintervals in the partition of
       !               (a,b),
       !              work(limit+1), ..., work(limit+last) contain the
       !               right end points,
       !              work(limit*2+1), ..., work(limit*2+last) contain
       !               the integral approximations over the subintervals,
       !              work(limit*3+1), ..., work(limit*3+last) contain
       !               the error estimates.
       !
       implicit none

       integer(int32) lenw
       integer(int32) limit

       real(real64) a
       real(real64) abserr
       real(real64) b
       real(real64) epsabs
       real(real64) epsrel
       integer(int32) ier
       integer(int32) iwork(limit)
       integer(int32) key
       integer(int32) last
       integer(int32) lvl
       integer(int32) l1
       integer(int32) l2
       integer(int32) l3
       integer(int32) neval
       real(real64) result
       real(real64) work(lenw)
       real(real64), external :: f

       !    interface
       !       function f(x)
       !         use iso_fortran_env
       !         real(real64), intent(in) :: x
       !         real(real64) :: f
       !       end function f
       !    end interface

       !
       !  check validity of lenw.
       !
       ier = 6
       neval = 0
       last = 0
       result = 0.0_real64
       abserr = 0.0_real64
       if(limit.lt.1.or.lenw.lt.limit*4) go to 10
       !
       !  prepare call for dqage.
       !
       l1 = limit+1
       l2 = limit+l1
       l3 = limit+l2

       call dqage(f,a,b,epsabs,epsrel,key,limit,result,abserr,neval, &
            ier,work(1),work(l1),work(l2),work(l3),iwork,last)
       !
       !  call error handler if necessary.
       !
       lvl = 0
10     continue

       if(ier.eq.6) lvl = 1
       if(ier.ne.0) call xerror('abnormal return from dqag ',26,ier,lvl)

       return
     end subroutine dqag
     subroutine dqelg ( n, epstab, result, abserr, res3la, nres )

       !*****************************************************************************80
       !
       !! DQELG carries out the Epsilon extrapolation algorithm.
       !
       !  Modified:
       !
       !    11 September 2015
       !
       !  Author:
       !
       !    Robert Piessens, Elise de Doncker
       !
       !***purpose  the routine determines the limit of a given sequence of
       !      approximations, by means of the epsilon algorithm of
       !      p.wynn. an estimate of the absolute error is also given.
       !      the condensed epsilon table is computed. only those
       !      elements needed for the computation of the next diagonal
       !      are preserved.
       !
       !  Parameters:
       !
       !        n      - integer ( kind = 4 )
       !                 epstab(n) contains the new element in the
       !                 first column of the epsilon table.
       !
       !        epstab - real ( kind = 8 )
       !                 vector of dimension 52 containing the elements
       !                 of the two lower diagonals of the triangular
       !                 epsilon table. the elements are numbered
       !                 starting at the right-hand corner of the
       !                 triangle.
       !
       !        result - real ( kind = 8 )
       !                 resulting approximation to the integral
       !
       !        abserr - real ( kind = 8 )
       !                 estimate of the absolute error computed from
       !                 result and the 3 previous results
       !
       !        res3la - real ( kind = 8 )
       !                 vector of dimension 3 containing the last 3
       !                 results
       !
       !        nres   - integer ( kind = 4 )
       !                 number of calls to the routine
       !                 (should be zero at first call)
       !
       !  Local Parameters:
       !
       !     e0     - the 4 elements on which the computation of a new
       !     e1       element in the epsilon table is based
       !     e2
       !     e3                 e0
       !                  e3    e1    new
       !                        e2
       !     newelm - number of elements to be computed in the new
       !              diagonal
       !     error  - error = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
       !     result - the element in the new diagonal with least value
       !              of error
       !
       !     machine dependent constants
       !
       !     epmach is the largest relative spacing.
       !     oflow is the largest positive magnitude.
       !     limexp is the maximum number of elements the epsilon
       !     table can contain. if this number is reached, the upper
       !     diagonal of the epsilon table is deleted.
       !
       implicit none

       real (real64) abserr,delta1,delta2,delta3, &
            epmach,epsinf,epstab,error,err1,err2,err3,e0,e1,e1abs,e2,e3, &
            oflow,res,result,res3la,ss,tol1,tol2,tol3
       integer (int32) i,ib,ib2,ie,indx,k1,k2,k3,limexp,n,newelm
       integer (int32) nres
       integer (int32) num
       dimension epstab(52),res3la(3)

       epmach = epsilon ( epmach )
       oflow = huge ( oflow )
       nres = nres+1
       abserr = oflow
       result = epstab(n)
       if(n.lt.3) go to 100
       limexp = 50
       epstab(n+2) = epstab(n)
       newelm = (n-1)/2
       epstab(n) = oflow
       num = n
       k1 = n

       do 40 i = 1,newelm

          k2 = k1-1
          k3 = k1-2
          res = epstab(k1+2)
          e0 = epstab(k3)
          e1 = epstab(k2)
          e2 = res
          e1abs =  abs ( e1)
          delta2 = e2-e1
          err2 =  abs ( delta2)
          tol2 =  max (  abs ( e2),e1abs)*epmach
          delta3 = e1 - e0
          err3 =  abs ( delta3)
          tol3 =  max ( e1abs, abs ( e0))*epmach
          if(err2.gt.tol2.or.err3.gt.tol3) go to 10
          !
          !  if e0, e1 and e2 are equal to machine accuracy, convergence is assumed.
          !
          result = res
          abserr = err2+err3
          go to 100
10        e3 = epstab(k1)
          epstab(k1) = e1
          delta1 = e1-e3
          err1 =  abs ( delta1)
          tol1 =  max ( e1abs, abs ( e3))*epmach
          !
          !  if two elements are very close to each other, omit
          !  a part of the table by adjusting the value of n
          !
          if(err1.le.tol1.or.err2.le.tol2.or.err3.le.tol3) go to 20
          ss = 0.1e1_real64/delta1+0.1e1_real64/delta2-0.1e1_real64/delta3
          epsinf =  abs ( ss*e1)
          !
          !  test to detect irregular behaviour in the table, and
          !  eventually omit a part of the table adjusting the value
          !  of n.
          !
          if(epsinf.gt.0.1e-3_real64) go to 30
20        n = i+i-1
          go to 50
          !
          !  compute a new element and eventually adjust
          !  the value of result.
          !
30        res = e1+0.1e1_real64/ss
          epstab(k1) = res
          k1 = k1-2
          error = err2 + abs ( res-e2 ) + err3

          if ( error .le. abserr ) then
             abserr = error
             result = res
          end if

40        continue
          !
          !  shift the table.
          !
50        if(n.eq.limexp) n = 2*(limexp/2)-1
          ib = 1
          if((num/2)*2.eq.num) ib = 2
          ie = newelm+1
          do i=1,ie
             ib2 = ib+2
             epstab(ib) = epstab(ib2)
             ib = ib2
          end do
          if(num.eq.n) go to 80
          indx = num-n+1
          do i = 1,n
             epstab(i)= epstab(indx)
             indx = indx+1
          end do
80        if(nres.ge.4) go to 90
          res3la(nres) = result
          abserr = oflow
          go to 100
          !
          !  compute error estimate
          !
90        abserr =  abs ( result-res3la(3))+ abs ( result-res3la(2)) &
               + abs ( result-res3la(1))
          res3la(1) = res3la(2)
          res3la(2) = res3la(3)
          res3la(3) = result
100       continue

          abserr =  max ( abserr, 0.5e1_real64*epmach* abs ( result))

          return
        end subroutine dqelg
        subroutine dqk15(f,a,b,result,abserr,resabs,resasc)

          !*****************************************************************************80
          !
          !! DQK15 carries out a 15 point Gauss-Kronrod quadrature rule.
          !
          !     the abscissae and weights are given for the interval (-1,1).
          !     because of symmetry only the positive abscissae and their
          !     corresponding weights are given.
          !
          !     xgk    - abscissae of the 15-point kronrod rule
          !              xgk(2), xgk(4), ...  abscissae of the 7-point
          !              gauss rule
          !              xgk(1), xgk(3), ...  abscissae which are optimally
          !              added to the 7-point gauss rule
          !
          !     wgk    - weights of the 15-point kronrod rule
          !
          !     wg     - weights of the 7-point gauss rule
          !
          !
          !   gauss quadrature weights and kronron quadrature abscissae and weights
          !   as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
          !   bell labs, nov. 1981.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  to compute i = integral of f over (a,b), with error
          !                     estimate
          !                 j = integral of abs(f) over (a,b)
          !  Parameters:
          !
          !      on entry
          !        f      - real(real64)
          !                 function subprogram defining the integrand
          !                 function f(x). the actual name for f needs to be
          !                 declared e x t e r n a l in the calling program.
          !
          !        a      - real(real64)
          !                 lower limit of integration
          !
          !        b      - real(real64)
          !                 upper limit of integration
          !
          !      on return
          !        result - real(real64)
          !                 approximation to the integral i
          !                 result is computed by applying the 15-point
          !                 kronrod rule (resk) obtained by optimal addition
          !                 of abscissae to the7-point gauss rule(resg).
          !
          !        abserr - real(real64)
          !                 estimate of the modulus of the absolute error,
          !                 which should not exceed abs(i-result)
          !
          !        resabs - real(real64)
          !                 approximation to the integral j
          !
          !        resasc - real(real64)
          !                 approximation to the integral of abs(f-i/(b-a))
          !                 over (a,b)
          !
          !  Local Parameters:
          !
          !     centr  - mid point of the interval
          !     hlgth  - half-length of the interval
          !     absc   - abscissa
          !     fval*  - function value
          !     resg   - result of the 7-point gauss formula
          !     resk   - result of the 15-point kronrod formula
          !     reskh  - approximation to the mean value of f over (a,b),
          !              i.e. to i/(b-a)
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,absc,abserr,b,centr,dhlgth, &
               epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc, &
               resg,resk,reskh,result,uflow,wg,wgk,xgk
          integer(int32) j,jtw,jtwm1
          external f
          dimension fv1(7),fv2(7),wg(4),wgk(8),xgk(8)

          data wg  (  1) / 0.129484966168869693270611432679082_real64 /
          data wg  (  2) / 0.279705391489276667901467771423780_real64 /
          data wg  (  3) / 0.381830050505118944950369775488975_real64 /
          data wg  (  4) / 0.417959183673469387755102040816327_real64 /

          data xgk (  1) / 0.991455371120812639206854697526329_real64 /
          data xgk (  2) / 0.949107912342758524526189684047851_real64 /
          data xgk (  3) / 0.864864423359769072789712788640926_real64 /
          data xgk (  4) / 0.741531185599394439863864773280788_real64 /
          data xgk (  5) / 0.586087235467691130294144838258730_real64 /
          data xgk (  6) / 0.405845151377397166906606412076961_real64 /
          data xgk (  7) / 0.207784955007898467600689403773245_real64 /
          data xgk (  8) / 0.000000000000000000000000000000000_real64 /

          data wgk (  1) / 0.022935322010529224963732008058970_real64 /
          data wgk (  2) / 0.063092092629978553290700663189204_real64 /
          data wgk (  3) / 0.104790010322250183839876322541518_real64 /
          data wgk (  4) / 0.140653259715525918745189590510238_real64 /
          data wgk (  5) / 0.169004726639267902826583426598550_real64 /
          data wgk (  6) / 0.190350578064785409913256402421014_real64 /
          data wgk (  7) / 0.204432940075298892414161999234649_real64 /
          data wgk (  8) / 0.209482141084727828012999174891714_real64 /

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          centr = 0.5_real64*(a+b)
          hlgth = 0.5_real64*(b-a)
          dhlgth =  abs ( hlgth)
          !
          !  compute the 15-point kronrod approximation to
          !  the integral, and estimate the absolute error.
          !
          fc = f(centr)
          resg = fc*wg(4)
          resk = fc*wgk(8)
          resabs =  abs ( resk)

          do j=1,3
             jtw = j*2
             absc = hlgth*xgk(jtw)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtw) = fval1
             fv2(jtw) = fval2
             fsum = fval1+fval2
             resg = resg+wg(j)*fsum
             resk = resk+wgk(jtw)*fsum
             resabs = resabs+wgk(jtw)*( abs ( fval1)+ abs ( fval2))
          end do

          do j = 1,4
             jtwm1 = j*2-1
             absc = hlgth*xgk(jtwm1)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtwm1) = fval1
             fv2(jtwm1) = fval2
             fsum = fval1+fval2
             resk = resk+wgk(jtwm1)*fsum
             resabs = resabs+wgk(jtwm1)*( abs ( fval1)+ abs ( fval2))
          end do

          reskh = resk*0.5_real64
          resasc = wgk(8)* abs ( fc-reskh)
          do j=1,7
             resasc = resasc+wgk(j)*( abs ( fv1(j)-reskh)+ abs ( fv2(j)-reskh))
          end do

          result = resk*hlgth
          resabs = resabs*dhlgth
          resasc = resasc*dhlgth
          abserr =  abs ( (resk-resg)*hlgth)
          if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) &
               abserr = resasc* min (0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
          if(resabs.gt.uflow/(0.5e2_real64*epmach)) abserr = max &
               ((epmach*0.5e2_real64)*resabs,abserr)

          return
        end subroutine dqk15
        subroutine dqk15i(f,boun,inf,a,b,result,abserr,resabs,resasc)

          !*****************************************************************************80
          !
          !! DQK15I applies a 15 point Gauss-Kronrod quadrature on an infinite interval.
          !
          !
          !     the abscissae and weights are supplied for the interval
          !     (-1,1).  because of symmetry only the positive abscissae and
          !     their corresponding weights are given.
          !
          !     xgk    - abscissae of the 15-point kronrod rule
          !              xgk(2), xgk(4), ... abscissae of the 7-point
          !              gauss rule
          !              xgk(1), xgk(3), ...  abscissae which are optimally
          !              added to the 7-point gauss rule
          !
          !     wgk    - weights of the 15-point kronrod rule
          !
          !     wg     - weights of the 7-point gauss rule, corresponding
          !              to the abscissae xgk(2), xgk(4), ...
          !              wg(1), wg(3), ... are set to zero.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  the original (infinite integration range is mapped
          !      onto the interval (0,1) and (a,b) is a part of (0,1).
          !      it is the purpose to compute
          !      i = integral of transformed integrand over (a,b),
          !      j = integral of abs(transformed integrand) over (a,b).
          !
          !  Parameters:
          !
          !      on entry
          !        f      - real(real64)
          !                 fuction subprogram defining the integrand
          !                 function f(x). the actual name for f needs to be
          !                 declared e x t e r n a l in the calling program.
          !
          !        boun   - real(real64)
          !                 finite bound of original integration
          !                 range (set to zero if inf = +2)
          !
          !        inf    - integer(int32)
          !                 if inf = -1, the original interval is
          !                             (-infinity,bound),
          !                 if inf = +1, the original interval is
          !                             (bound,+infinity),
          !                 if inf = +2, the original interval is
          !                             (-infinity,+infinity) and
          !                 the integral is computed as the sum of two
          !                 integrals, one over (-infinity,0) and one over
          !                 (0,+infinity).
          !
          !        a      - real(real64)
          !                 lower limit for integration over subrange
          !                 of (0,1)
          !
          !        b      - real(real64)
          !                 upper limit for integration over subrange
          !                 of (0,1)
          !
          !      on return
          !        result - real(real64)
          !                 approximation to the integral i
          !                 result is computed by applying the 15-point
          !                 kronrod rule(resk) obtained by optimal addition
          !                 of abscissae to the 7-point gauss rule(resg).
          !
          !        abserr - real(real64)
          !                 estimate of the modulus of the absolute error,
          !                 which should equal or exceed abs(i-result)
          !
          !        resabs - real(real64)
          !                 approximation to the integral j
          !
          !        resasc - real(real64)
          !                 approximation to the integral of
          !                 abs((transformed integrand)-i/(b-a)) over (a,b)
          !
          !  Local Parameters:
          !
          !     centr  - mid point of the interval
          !     hlgth  - half-length of the interval
          !     absc*  - abscissa
          !     tabsc* - transformed abscissa
          !     fval*  - function value
          !     resg   - result of the 7-point gauss formula
          !     resk   - result of the 15-point kronrod formula
          !     reskh  - approximation to the mean value of the transformed
          !              integrand over (a,b), i.e. to i/(b-a)
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,absc,absc1,absc2,abserr,b,boun,centr,dinf, &
               epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth, &
               resabs,resasc,resg,resk,reskh,result,tabsc1,tabsc2,uflow,wg,wgk, &
               xgk
          integer(int32) inf,j
          external f
          dimension fv1(7),fv2(7),xgk(8),wgk(8),wg(8)

          data wg(1) / 0.0_real64 /
          data wg(2) / 0.129484966168869693270611432679082_real64 /
          data wg(3) / 0.0_real64 /
          data wg(4) / 0.279705391489276667901467771423780_real64 /
          data wg(5) / 0.0_real64 /
          data wg(6) / 0.381830050505118944950369775488975_real64 /
          data wg(7) / 0.0_real64 /
          data wg(8) / 0.417959183673469387755102040816327_real64 /

          data xgk(1) / 0.991455371120812639206854697526329_real64 /
          data xgk(2) / 0.949107912342758524526189684047851_real64 /
          data xgk(3) / 0.864864423359769072789712788640926_real64 /
          data xgk(4) / 0.741531185599394439863864773280788_real64 /
          data xgk(5) / 0.586087235467691130294144838258730_real64 /
          data xgk(6) / 0.405845151377397166906606412076961_real64 /
          data xgk(7) / 0.207784955007898467600689403773245_real64 /
          data xgk(8) / 0.000000000000000000000000000000000_real64 /

          data wgk(1) / 0.022935322010529224963732008058970_real64 /
          data wgk(2) / 0.063092092629978553290700663189204_real64 /
          data wgk(3) / 0.104790010322250183839876322541518_real64 /
          data wgk(4) / 0.140653259715525918745189590510238_real64 /
          data wgk(5) / 0.169004726639267902826583426598550_real64 /
          data wgk(6) / 0.190350578064785409913256402421014_real64 /
          data wgk(7) / 0.204432940075298892414161999234649_real64 /
          data wgk(8) / 0.209482141084727828012999174891714_real64 /

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          dinf = min ( 1, inf )
          centr = 0.5_real64*(a+b)
          hlgth = 0.5_real64*(b-a)
          tabsc1 = boun+dinf*(0.1e1_real64-centr)/centr
          fval1 = f(tabsc1)
          if(inf.eq.2) fval1 = fval1+f(-tabsc1)
          fc = (fval1/centr)/centr
          !
          !  compute the 15-point kronrod approximation to
          !  the integral, and estimate the error.
          !
          resg = wg(8)*fc
          resk = wgk(8)*fc
          resabs =  abs ( resk)

          do j=1,7
             absc = hlgth*xgk(j)
             absc1 = centr-absc
             absc2 = centr+absc
             tabsc1 = boun+dinf*(0.1e1_real64-absc1)/absc1
             tabsc2 = boun+dinf*(0.1e1_real64-absc2)/absc2
             fval1 = f(tabsc1)
             fval2 = f(tabsc2)
             if(inf.eq.2) fval1 = fval1+f(-tabsc1)
             if(inf.eq.2) fval2 = fval2+f(-tabsc2)
             fval1 = (fval1/absc1)/absc1
             fval2 = (fval2/absc2)/absc2
             fv1(j) = fval1
             fv2(j) = fval2
             fsum = fval1+fval2
             resg = resg+wg(j)*fsum
             resk = resk+wgk(j)*fsum
             resabs = resabs+wgk(j)*( abs ( fval1)+ abs ( fval2))
          end do

          reskh = resk*0.5_real64
          resasc = wgk(8)* abs ( fc-reskh)

          do j=1,7
             resasc = resasc+wgk(j)*( abs ( fv1(j)-reskh)+ abs ( fv2(j)-reskh))
          end do

          result = resk*hlgth
          resasc = resasc*hlgth
          resabs = resabs*hlgth
          abserr =  abs ( (resk-resg)*hlgth)
          if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) abserr = resasc* &
               min (0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
          if(resabs.gt.uflow/(0.5e2_real64*epmach)) abserr = max &
               ((epmach*0.5e2_real64)*resabs,abserr)

          return
        end subroutine dqk15i
        subroutine dqk15w(f,w,p1,p2,p3,p4,kp,a,b,result,abserr, resabs,resasc)

          !*****************************************************************************80
          !
          !! DQK15W applies a 15 point Gauss-Kronrod rule for a weighted integrand.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  to compute i = integral of f*w over (a,b), with error
          !                     estimate
          !                 j = integral of abs(f*w) over (a,b)
          !
          !  Parameters:
          !
          !       on entry
          !        f      - real(real64)
          !                 function subprogram defining the integrand
          !                 function f(x). the actual name for f needs to be
          !                 declared e x t e r n a l in the driver program.
          !
          !        w      - real(real64)
          !                 function subprogram defining the integrand
          !                 weight function w(x). the actual name for w
          !                 needs to be declared e x t e r n a l in the
          !                 calling program.
          !
          !        p1, p2, p3, p4 - real(real64)
          !                 parameters in the weight function
          !
          !        kp     - integer(int32)
          !                 key for indicating the type of weight function
          !
          !        a      - real(real64)
          !                 lower limit of integration
          !
          !        b      - real(real64)
          !                 upper limit of integration
          !
          !      on return
          !        result - real(real64)
          !                 approximation to the integral i
          !                 result is computed by applying the 15-point
          !                 kronrod rule (resk) obtained by optimal addition
          !                 of abscissae to the 7-point gauss rule (resg).
          !
          !        abserr - real(real64)
          !                 estimate of the modulus of the absolute error,
          !                 which should equal or exceed abs(i-result)
          !
          !        resabs - real(real64)
          !                 approximation to the integral of abs(f)
          !
          !        resasc - real(real64)
          !                 approximation to the integral of abs(f-i/(b-a))
          !
          !  Local Parameters:
          !
          !     the abscissae and weights are given for the interval (-1,1).
          !     because of symmetry only the positive abscissae and their
          !     corresponding weights are given.
          !
          !     xgk    - abscissae of the 15-point gauss-kronrod rule
          !              xgk(2), xgk(4), ... abscissae of the 7-point
          !              gauss rule
          !              xgk(1), xgk(3), ... abscissae which are optimally
          !              added to the 7-point gauss rule
          !
          !     wgk    - weights of the 15-point gauss-kronrod rule
          !
          !     wg     - weights of the 7-point gauss rule
          !
          !     centr  - mid point of the interval
          !     hlgth  - half-length of the interval
          !     absc*  - abscissa
          !     fval*  - function value
          !     resg   - result of the 7-point gauss formula
          !     resk   - result of the 15-point kronrod formula
          !     reskh  - approximation to the mean value of f*w over (a,b),
          !              i.e. to i/(b-a)
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,absc,absc1,absc2,abserr,b,centr,dhlgth, &
               epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth, &
               p1,p2,p3,p4,resabs,resasc,resg,resk,reskh,result,uflow,w,wg,wgk, &
               xgk
          integer(int32) j,jtw,jtwm1,kp
          external f,w

          dimension fv1(7),fv2(7),xgk(8),wgk(8),wg(4)

          data xgk(1),xgk(2),xgk(3),xgk(4),xgk(5),xgk(6),xgk(7),xgk(8)/ &
               0.9914553711208126_real64,     0.9491079123427585_real64, &
               0.8648644233597691_real64,     0.7415311855993944_real64, &
               0.5860872354676911_real64,     0.4058451513773972_real64, &
               0.2077849550078985_real64,     0.0000000000000000_real64/

          data wgk(1),wgk(2),wgk(3),wgk(4),wgk(5),wgk(6),wgk(7),wgk(8)/ &
               0.2293532201052922e-1_real64,     0.6309209262997855e-1_real64, &
               0.1047900103222502_real64,     0.1406532597155259_real64, &
               0.1690047266392679_real64,     0.1903505780647854_real64, &
               0.2044329400752989_real64,     0.2094821410847278_real64/

          data wg(1),wg(2),wg(3),wg(4)/ &
               0.1294849661688697_real64,    0.2797053914892767_real64, &
               0.3818300505051889_real64,    0.4179591836734694_real64/

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          centr = 0.5_real64*(a+b)
          hlgth = 0.5_real64*(b-a)
          dhlgth =  abs ( hlgth)
          !
          !  compute the 15-point kronrod approximation to the
          !  integral, and estimate the error.
          !
          fc = f(centr)*w(centr,p1,p2,p3,p4,kp)
          resg = wg(4)*fc
          resk = wgk(8)*fc
          resabs =  abs ( resk)

          do j=1,3
             jtw = j*2
             absc = hlgth*xgk(jtw)
             absc1 = centr-absc
             absc2 = centr+absc
             fval1 = f(absc1)*w(absc1,p1,p2,p3,p4,kp)
             fval2 = f(absc2)*w(absc2,p1,p2,p3,p4,kp)
             fv1(jtw) = fval1
             fv2(jtw) = fval2
             fsum = fval1+fval2
             resg = resg+wg(j)*fsum
             resk = resk+wgk(jtw)*fsum
             resabs = resabs+wgk(jtw)*( abs ( fval1)+ abs ( fval2))
          end do

          do j=1,4
             jtwm1 = j*2-1
             absc = hlgth*xgk(jtwm1)
             absc1 = centr-absc
             absc2 = centr+absc
             fval1 = f(absc1)*w(absc1,p1,p2,p3,p4,kp)
             fval2 = f(absc2)*w(absc2,p1,p2,p3,p4,kp)
             fv1(jtwm1) = fval1
             fv2(jtwm1) = fval2
             fsum = fval1+fval2
             resk = resk+wgk(jtwm1)*fsum
             resabs = resabs+wgk(jtwm1)*( abs ( fval1)+ abs ( fval2))
          end do

          reskh = resk*0.5_real64
          resasc = wgk(8)* abs ( fc-reskh)

          do j=1,7
             resasc = resasc+wgk(j)*( abs ( fv1(j)-reskh)+ abs ( fv2(j)-reskh))
          end do

          result = resk*hlgth
          resabs = resabs*dhlgth
          resasc = resasc*dhlgth
          abserr =  abs ( (resk-resg)*hlgth)
          if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) &
               abserr = resasc* min (0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
          if(resabs.gt.uflow/(0.5e2_real64*epmach)) abserr =  max ( (epmach* &
               0.5e2_real64)*resabs,abserr)

          return
        end subroutine dqk15w
        subroutine dqk21(f,a,b,result,abserr,resabs,resasc)

          !*****************************************************************************80
          !
          !! DQK21 carries out a 21 point Gauss-Kronrod quadrature rule.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  to compute i = integral of f over (a,b), with error
          !                     estimate
          !                 j = integral of abs(f) over (a,b)
          !
          !  Parameters:
          !
          !      on entry
          !        f      - real(real64)
          !                 function subprogram defining the integrand
          !                 function f(x). the actual name for f needs to be
          !                 declared e x t e r n a l in the driver program.
          !
          !        a      - real(real64)
          !                 lower limit of integration
          !
          !        b      - real(real64)
          !                 upper limit of integration
          !
          !      on return
          !        result - real(real64)
          !                 approximation to the integral i
          !                 result is computed by applying the 21-point
          !                 kronrod rule (resk) obtained by optimal addition
          !                 of abscissae to the 10-point gauss rule (resg).
          !
          !        abserr - real(real64)
          !                 estimate of the modulus of the absolute error,
          !                 which should not exceed abs(i-result)
          !
          !        resabs - real(real64)
          !                 approximation to the integral j
          !
          !        resasc - real(real64)
          !                 approximation to the integral of abs(f-i/(b-a))
          !                 over (a,b)
          !
          !  Local Parameters:
          !
          !
          !     the abscissae and weights are given for the interval (-1,1).
          !     because of symmetry only the positive abscissae and their
          !     corresponding weights are given.
          !
          !     xgk    - abscissae of the 21-point kronrod rule
          !              xgk(2), xgk(4), ...  abscissae of the 10-point
          !              gauss rule
          !              xgk(1), xgk(3), ...  abscissae which are optimally
          !              added to the 10-point gauss rule
          !
          !     wgk    - weights of the 21-point kronrod rule
          !
          !     wg     - weights of the 10-point gauss rule
          !
          !
          ! gauss quadrature weights and kronron quadrature abscissae and weights
          ! as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
          ! bell labs, nov. 1981.
          !
          !     centr  - mid point of the interval
          !     hlgth  - half-length of the interval
          !     absc   - abscissa
          !     fval*  - function value
          !     resg   - result of the 10-point gauss formula
          !     resk   - result of the 21-point kronrod formula
          !     reskh  - approximation to the mean value of f over (a,b),
          !              i.e. to i/(b-a)
          !
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,absc,abserr,b,centr,dhlgth, &
               epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc, &
               resg,resk,reskh,result,uflow,wg,wgk,xgk
          integer(int32) j,jtw,jtwm1
          external f
          dimension fv1(10),fv2(10),wg(5),wgk(11),xgk(11)

          data wg  (  1) / 0.066671344308688137593568809893332_real64 /
          data wg  (  2) / 0.149451349150580593145776339657697_real64 /
          data wg  (  3) / 0.219086362515982043995534934228163_real64 /
          data wg  (  4) / 0.269266719309996355091226921569469_real64 /
          data wg  (  5) / 0.295524224714752870173892994651338_real64 /

          data xgk (  1) / 0.995657163025808080735527280689003_real64 /
          data xgk (  2) / 0.973906528517171720077964012084452_real64 /
          data xgk (  3) / 0.930157491355708226001207180059508_real64 /
          data xgk (  4) / 0.865063366688984510732096688423493_real64 /
          data xgk (  5) / 0.780817726586416897063717578345042_real64 /
          data xgk (  6) / 0.679409568299024406234327365114874_real64 /
          data xgk (  7) / 0.562757134668604683339000099272694_real64 /
          data xgk (  8) / 0.433395394129247190799265943165784_real64 /
          data xgk (  9) / 0.294392862701460198131126603103866_real64 /
          data xgk ( 10) / 0.148874338981631210884826001129720_real64 /
          data xgk ( 11) / 0.000000000000000000000000000000000_real64 /

          data wgk (  1) / 0.011694638867371874278064396062192_real64 /
          data wgk (  2) / 0.032558162307964727478818972459390_real64 /
          data wgk (  3) / 0.054755896574351996031381300244580_real64 /
          data wgk (  4) / 0.075039674810919952767043140916190_real64 /
          data wgk (  5) / 0.093125454583697605535065465083366_real64 /
          data wgk (  6) / 0.109387158802297641899210590325805_real64 /
          data wgk (  7) / 0.123491976262065851077958109831074_real64 /
          data wgk (  8) / 0.134709217311473325928054001771707_real64 /
          data wgk (  9) / 0.142775938577060080797094273138717_real64 /
          data wgk ( 10) / 0.147739104901338491374841515972068_real64 /
          data wgk ( 11) / 0.149445554002916905664936468389821_real64 /

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          centr = 0.5_real64*(a+b)
          hlgth = 0.5_real64*(b-a)
          dhlgth =  abs ( hlgth)
          !
          !  compute the 21-point kronrod approximation to
          !  the integral, and estimate the absolute error.
          !
          resg = 0.0_real64
          fc = f(centr)
          resk = wgk(11)*fc
          resabs =  abs ( resk)
          do j=1,5
             jtw = 2*j
             absc = hlgth*xgk(jtw)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtw) = fval1
             fv2(jtw) = fval2
             fsum = fval1+fval2
             resg = resg+wg(j)*fsum
             resk = resk+wgk(jtw)*fsum
             resabs = resabs+wgk(jtw)*( abs ( fval1)+ abs ( fval2))
          end do

          do j = 1,5
             jtwm1 = 2*j-1
             absc = hlgth*xgk(jtwm1)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtwm1) = fval1
             fv2(jtwm1) = fval2
             fsum = fval1+fval2
             resk = resk+wgk(jtwm1)*fsum
             resabs = resabs+wgk(jtwm1)*( abs ( fval1)+ abs ( fval2))
          end do

          reskh = resk*0.5_real64
          resasc = wgk(11)* abs ( fc-reskh)

          do j=1,10
             resasc = resasc+wgk(j)*( abs ( fv1(j)-reskh)+ abs ( fv2(j)-reskh))
          end do

          result = resk*hlgth
          resabs = resabs*dhlgth
          resasc = resasc*dhlgth
          abserr =  abs ( (resk-resg)*hlgth)
          if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) &
               abserr = resasc*min(0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
          if(resabs.gt.uflow/(0.5e2_real64*epmach)) abserr = max &
               ((epmach*0.5e2_real64)*resabs,abserr)

          return
        end subroutine dqk21
        subroutine dqk31(f,a,b,result,abserr,resabs,resasc)

          !*****************************************************************************80
          !
          !! DQK31 carries out a 31 point Gauss-Kronrod quadrature rule.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  to compute i = integral of f over (a,b) with error
          !                     estimate
          !                 j = integral of abs(f) over (a,b)
          !
          !  Parameters:
          !
          !      on entry
          !        f      - real(real64)
          !                 function subprogram defining the integrand
          !                 function f(x). the actual name for f needs to be
          !                 declared e x t e r n a l in the calling program.
          !
          !        a      - real(real64)
          !                 lower limit of integration
          !
          !        b      - real(real64)
          !                 upper limit of integration
          !
          !      on return
          !        result - real(real64)
          !                 approximation to the integral i
          !                 result is computed by applying the 31-point
          !                 gauss-kronrod rule (resk), obtained by optimal
          !                 addition of abscissae to the 15-point gauss
          !                 rule (resg).
          !
          !        abserr - double precison
          !                 estimate of the modulus of the modulus,
          !                 which should not exceed abs(i-result)
          !
          !        resabs - real(real64)
          !                 approximation to the integral j
          !
          !        resasc - real(real64)
          !                 approximation to the integral of abs(f-i/(b-a))
          !                 over (a,b)
          !
          !  Local Parameters:
          !
          !
          !     the abscissae and weights are given for the interval (-1,1).
          !     because of symmetry only the positive abscissae and their
          !     corresponding weights are given.
          !
          !     xgk    - abscissae of the 31-point kronrod rule
          !              xgk(2), xgk(4), ...  abscissae of the 15-point
          !              gauss rule
          !              xgk(1), xgk(3), ...  abscissae which are optimally
          !              added to the 15-point gauss rule
          !
          !     wgk    - weights of the 31-point kronrod rule
          !
          !     wg     - weights of the 15-point gauss rule
          !
          !
          ! gauss quadrature weights and kronron quadrature abscissae and weights
          ! as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
          ! bell labs, nov. 1981.
          !
          !     centr  - mid point of the interval
          !     hlgth  - half-length of the interval
          !     absc   - abscissa
          !     fval*  - function value
          !     resg   - result of the 15-point gauss formula
          !     resk   - result of the 31-point kronrod formula
          !     reskh  - approximation to the mean value of f over (a,b),
          !              i.e. to i/(b-a)
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,absc,abserr,b,centr,dhlgth, &
               epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc, &
               resg,resk,reskh,result,uflow,wg,wgk,xgk
          integer(int32) j,jtw,jtwm1
          external f

          dimension fv1(15),fv2(15),xgk(16),wgk(16),wg(8)

          data wg  (  1) / 0.030753241996117268354628393577204_real64 /
          data wg  (  2) / 0.070366047488108124709267416450667_real64 /
          data wg  (  3) / 0.107159220467171935011869546685869_real64 /
          data wg  (  4) / 0.139570677926154314447804794511028_real64 /
          data wg  (  5) / 0.166269205816993933553200860481209_real64 /
          data wg  (  6) / 0.186161000015562211026800561866423_real64 /
          data wg  (  7) / 0.198431485327111576456118326443839_real64 /
          data wg  (  8) / 0.202578241925561272880620199967519_real64 /

          data xgk (  1) / 0.998002298693397060285172840152271_real64 /
          data xgk (  2) / 0.987992518020485428489565718586613_real64 /
          data xgk (  3) / 0.967739075679139134257347978784337_real64 /
          data xgk (  4) / 0.937273392400705904307758947710209_real64 /
          data xgk (  5) / 0.897264532344081900882509656454496_real64 /
          data xgk (  6) / 0.848206583410427216200648320774217_real64 /
          data xgk (  7) / 0.790418501442465932967649294817947_real64 /
          data xgk (  8) / 0.724417731360170047416186054613938_real64 /
          data xgk (  9) / 0.650996741297416970533735895313275_real64 /
          data xgk ( 10) / 0.570972172608538847537226737253911_real64 /
          data xgk ( 11) / 0.485081863640239680693655740232351_real64 /
          data xgk ( 12) / 0.394151347077563369897207370981045_real64 /
          data xgk ( 13) / 0.299180007153168812166780024266389_real64 /
          data xgk ( 14) / 0.201194093997434522300628303394596_real64 /
          data xgk ( 15) / 0.101142066918717499027074231447392_real64 /
          data xgk ( 16) / 0.000000000000000000000000000000000_real64 /

          data wgk (  1) / 0.005377479872923348987792051430128_real64 /
          data wgk (  2) / 0.015007947329316122538374763075807_real64 /
          data wgk (  3) / 0.025460847326715320186874001019653_real64 /
          data wgk (  4) / 0.035346360791375846222037948478360_real64 /
          data wgk (  5) / 0.044589751324764876608227299373280_real64 /
          data wgk (  6) / 0.053481524690928087265343147239430_real64 /
          data wgk (  7) / 0.062009567800670640285139230960803_real64 /
          data wgk (  8) / 0.069854121318728258709520077099147_real64 /
          data wgk (  9) / 0.076849680757720378894432777482659_real64 /
          data wgk ( 10) / 0.083080502823133021038289247286104_real64 /
          data wgk ( 11) / 0.088564443056211770647275443693774_real64 /
          data wgk ( 12) / 0.093126598170825321225486872747346_real64 /
          data wgk ( 13) / 0.096642726983623678505179907627589_real64 /
          data wgk ( 14) / 0.099173598721791959332393173484603_real64 /
          data wgk ( 15) / 0.100769845523875595044946662617570_real64 /
          data wgk ( 16) / 0.101330007014791549017374792767493_real64 /

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          centr = 0.5_real64*(a+b)
          hlgth = 0.5_real64*(b-a)
          dhlgth =  abs ( hlgth)
          !
          !  compute the 31-point kronrod approximation to
          !  the integral, and estimate the absolute error.
          !
          fc = f(centr)
          resg = wg(8)*fc
          resk = wgk(16)*fc
          resabs =  abs ( resk)

          do j=1,7
             jtw = j*2
             absc = hlgth*xgk(jtw)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtw) = fval1
             fv2(jtw) = fval2
             fsum = fval1+fval2
             resg = resg+wg(j)*fsum
             resk = resk+wgk(jtw)*fsum
             resabs = resabs+wgk(jtw)*( abs ( fval1)+ abs ( fval2))
          end do

          do j = 1,8
             jtwm1 = j*2-1
             absc = hlgth*xgk(jtwm1)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtwm1) = fval1
             fv2(jtwm1) = fval2
             fsum = fval1+fval2
             resk = resk+wgk(jtwm1)*fsum
             resabs = resabs+wgk(jtwm1)*( abs ( fval1)+ abs ( fval2))
          end do

          reskh = resk*0.5_real64
          resasc = wgk(16)* abs ( fc-reskh)

          do j=1,15
             resasc = resasc+wgk(j)*( abs ( fv1(j)-reskh)+ abs ( fv2(j)-reskh))
          end do

          result = resk*hlgth
          resabs = resabs*dhlgth
          resasc = resasc*dhlgth
          abserr =  abs ( (resk-resg)*hlgth)
          if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) &
               abserr = resasc* min (0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
          if(resabs.gt.uflow/(0.5e2_real64*epmach)) abserr = max &
               ((epmach*0.5e2_real64)*resabs,abserr)

          return
        end subroutine dqk31
        subroutine dqk41 ( f, a, b, result, abserr, resabs, resasc )

          !*****************************************************************************80
          !
          !! DQK41 carries out a 41 point Gauss-Kronrod quadrature rule.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  to compute i = integral of f over (a,b), with error
          !                     estimate
          !                 j = integral of abs(f) over (a,b)
          !
          !  Parameters:
          !
          !      on entry
          !        f      - real(real64)
          !                 function subprogram defining the integrand
          !                 function f(x). the actual name for f needs to be
          !                 declared e x t e r n a l in the calling program.
          !
          !        a      - real(real64)
          !                 lower limit of integration
          !
          !        b      - real(real64)
          !                 upper limit of integration
          !
          !      on return
          !        result - real(real64)
          !                 approximation to the integral i
          !                 result is computed by applying the 41-point
          !                 gauss-kronrod rule (resk) obtained by optimal
          !                 addition of abscissae to the 20-point gauss
          !                 rule (resg).
          !
          !        abserr - real(real64)
          !                 estimate of the modulus of the absolute error,
          !                 which should not exceed abs(i-result)
          !
          !        resabs - real(real64)
          !                 approximation to the integral j
          !
          !        resasc - real(real64)
          !                 approximation to the integal of abs(f-i/(b-a))
          !                 over (a,b)
          !
          !  Local Parameters:
          !
          !
          !     the abscissae and weights are given for the interval (-1,1).
          !     because of symmetry only the positive abscissae and their
          !     corresponding weights are given.
          !
          !     xgk    - abscissae of the 41-point gauss-kronrod rule
          !              xgk(2), xgk(4), ...  abscissae of the 20-point
          !              gauss rule
          !              xgk(1), xgk(3), ...  abscissae which are optimally
          !              added to the 20-point gauss rule
          !
          !     wgk    - weights of the 41-point gauss-kronrod rule
          !
          !     wg     - weights of the 20-point gauss rule
          !
          !
          ! gauss quadrature weights and kronron quadrature abscissae and weights
          ! as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
          ! bell labs, nov. 1981.
          !
          !     centr  - mid point of the interval
          !     hlgth  - half-length of the interval
          !     absc   - abscissa
          !     fval*  - function value
          !     resg   - result of the 20-point gauss formula
          !     resk   - result of the 41-point kronrod formula
          !     reskh  - approximation to mean value of f over (a,b), i.e.
          !              to i/(b-a)
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,absc,abserr,b,centr,dhlgth, &
               epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc, &
               resg,resk,reskh,result,uflow,wg,wgk,xgk
          integer(int32) j,jtw,jtwm1
          external f

          dimension fv1(20),fv2(20),xgk(21),wgk(21),wg(10)

          data wg  (  1) / 0.017614007139152118311861962351853_real64 /
          data wg  (  2) / 0.040601429800386941331039952274932_real64 /
          data wg  (  3) / 0.062672048334109063569506535187042_real64 /
          data wg  (  4) / 0.083276741576704748724758143222046_real64 /
          data wg  (  5) / 0.101930119817240435036750135480350_real64 /
          data wg  (  6) / 0.118194531961518417312377377711382_real64 /
          data wg  (  7) / 0.131688638449176626898494499748163_real64 /
          data wg  (  8) / 0.142096109318382051329298325067165_real64 /
          data wg  (  9) / 0.149172986472603746787828737001969_real64 /
          data wg  ( 10) / 0.152753387130725850698084331955098_real64 /

          data xgk (  1) / 0.998859031588277663838315576545863_real64 /
          data xgk (  2) / 0.993128599185094924786122388471320_real64 /
          data xgk (  3) / 0.981507877450250259193342994720217_real64 /
          data xgk (  4) / 0.963971927277913791267666131197277_real64 /
          data xgk (  5) / 0.940822633831754753519982722212443_real64 /
          data xgk (  6) / 0.912234428251325905867752441203298_real64 /
          data xgk (  7) / 0.878276811252281976077442995113078_real64 /
          data xgk (  8) / 0.839116971822218823394529061701521_real64 /
          data xgk (  9) / 0.795041428837551198350638833272788_real64 /
          data xgk ( 10) / 0.746331906460150792614305070355642_real64 /
          data xgk ( 11) / 0.693237656334751384805490711845932_real64 /
          data xgk ( 12) / 0.636053680726515025452836696226286_real64 /
          data xgk ( 13) / 0.575140446819710315342946036586425_real64 /
          data xgk ( 14) / 0.510867001950827098004364050955251_real64 /
          data xgk ( 15) / 0.443593175238725103199992213492640_real64 /
          data xgk ( 16) / 0.373706088715419560672548177024927_real64 /
          data xgk ( 17) / 0.301627868114913004320555356858592_real64 /
          data xgk ( 18) / 0.227785851141645078080496195368575_real64 /
          data xgk ( 19) / 0.152605465240922675505220241022678_real64 /
          data xgk ( 20) / 0.076526521133497333754640409398838_real64 /
          data xgk ( 21) / 0.000000000000000000000000000000000_real64 /

          data wgk (  1) / 0.003073583718520531501218293246031_real64 /
          data wgk (  2) / 0.008600269855642942198661787950102_real64 /
          data wgk (  3) / 0.014626169256971252983787960308868_real64 /
          data wgk (  4) / 0.020388373461266523598010231432755_real64 /
          data wgk (  5) / 0.025882133604951158834505067096153_real64 /
          data wgk (  6) / 0.031287306777032798958543119323801_real64 /
          data wgk (  7) / 0.036600169758200798030557240707211_real64 /
          data wgk (  8) / 0.041668873327973686263788305936895_real64 /
          data wgk (  9) / 0.046434821867497674720231880926108_real64 /
          data wgk ( 10) / 0.050944573923728691932707670050345_real64 /
          data wgk ( 11) / 0.055195105348285994744832372419777_real64 /
          data wgk ( 12) / 0.059111400880639572374967220648594_real64 /
          data wgk ( 13) / 0.062653237554781168025870122174255_real64 /
          data wgk ( 14) / 0.065834597133618422111563556969398_real64 /
          data wgk ( 15) / 0.068648672928521619345623411885368_real64 /
          data wgk ( 16) / 0.071054423553444068305790361723210_real64 /
          data wgk ( 17) / 0.073030690332786667495189417658913_real64 /
          data wgk ( 18) / 0.074582875400499188986581418362488_real64 /
          data wgk ( 19) / 0.075704497684556674659542775376617_real64 /
          data wgk ( 20) / 0.076377867672080736705502835038061_real64 /
          data wgk ( 21) / 0.076600711917999656445049901530102_real64 /

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          centr = 0.5_real64*(a+b)
          hlgth = 0.5_real64*(b-a)
          dhlgth =  abs ( hlgth)
          !
          !  compute the 41-point gauss-kronrod approximation to
          !  the integral, and estimate the absolute error.
          !
          resg = 0.0_real64
          fc = f(centr)
          resk = wgk(21)*fc
          resabs =  abs ( resk)

          do j=1,10
             jtw = j*2
             absc = hlgth*xgk(jtw)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtw) = fval1
             fv2(jtw) = fval2
             fsum = fval1+fval2
             resg = resg+wg(j)*fsum
             resk = resk+wgk(jtw)*fsum
             resabs = resabs+wgk(jtw)*( abs ( fval1)+ abs ( fval2))
          end do

          do j = 1,10
             jtwm1 = j*2-1
             absc = hlgth*xgk(jtwm1)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtwm1) = fval1
             fv2(jtwm1) = fval2
             fsum = fval1+fval2
             resk = resk+wgk(jtwm1)*fsum
             resabs = resabs+wgk(jtwm1)*( abs ( fval1)+ abs ( fval2))
          end do

          reskh = resk*0.5_real64
          resasc = wgk(21)* abs ( fc-reskh)

          do j=1,20
             resasc = resasc+wgk(j)*( abs ( fv1(j)-reskh)+ abs ( fv2(j)-reskh))
          end do

          result = resk*hlgth
          resabs = resabs*dhlgth
          resasc = resasc*dhlgth
          abserr =  abs ( (resk-resg)*hlgth)
          if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) &
               abserr = resasc* min (0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
          if(resabs.gt.uflow/(0.5e2_real64*epmach)) abserr = max &
               ((epmach*0.5e2_real64)*resabs,abserr)

          return
        end subroutine dqk41
        subroutine dqk51(f,a,b,result,abserr,resabs,resasc)

          !*****************************************************************************80
          !
          !! DQK51 carries out a 51 point Gauss-Kronrod quadrature rule.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  to compute i = integral of f over (a,b) with error
          !                     estimate
          !                 j = integral of abs(f) over (a,b)
          !
          !  Parameters:
          !
          !      on entry
          !        f      - real(real64)
          !                 function defining the integrand
          !                 function f(x). the actual name for f needs to be
          !                 declared e x t e r n a l in the calling program.
          !
          !        a      - real(real64)
          !                 lower limit of integration
          !
          !        b      - real(real64)
          !                 upper limit of integration
          !
          !      on return
          !        result - real(real64)
          !                 approximation to the integral i
          !                 result is computed by applying the 51-point
          !                 kronrod rule (resk) obtained by optimal addition
          !                 of abscissae to the 25-point gauss rule (resg).
          !
          !        abserr - real(real64)
          !                 estimate of the modulus of the absolute error,
          !                 which should not exceed abs(i-result)
          !
          !        resabs - real(real64)
          !                 approximation to the integral j
          !
          !        resasc - real(real64)
          !                 approximation to the integral of abs(f-i/(b-a))
          !                 over (a,b)
          !
          !  Local Parameters:
          !
          !     the abscissae and weights are given for the interval (-1,1).
          !     because of symmetry only the positive abscissae and their
          !     corresponding weights are given.
          !
          !     xgk    - abscissae of the 51-point kronrod rule
          !              xgk(2), xgk(4), ...  abscissae of the 25-point
          !              gauss rule
          !              xgk(1), xgk(3), ...  abscissae which are optimally
          !              added to the 25-point gauss rule
          !
          !     wgk    - weights of the 51-point kronrod rule
          !
          !     wg     - weights of the 25-point gauss rule
          !
          ! gauss quadrature weights and kronron quadrature abscissae and weights
          ! as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
          ! bell labs, nov. 1981.
          !
          !     centr  - mid point of the interval
          !     hlgth  - half-length of the interval
          !     absc   - abscissa
          !     fval*  - function value
          !     resg   - result of the 25-point gauss formula
          !     resk   - result of the 51-point kronrod formula
          !     reskh  - approximation to the mean value of f over (a,b),
          !              i.e. to i/(b-a)
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,absc,abserr,b,centr,dhlgth, &
               epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc, &
               resg,resk,reskh,result,uflow,wg,wgk,xgk
          integer(int32) j,jtw,jtwm1
          external f

          dimension fv1(25),fv2(25),xgk(26),wgk(26),wg(13)

          data wg  (  1) / 0.011393798501026287947902964113235_real64 /
          data wg  (  2) / 0.026354986615032137261901815295299_real64 /
          data wg  (  3) / 0.040939156701306312655623487711646_real64 /
          data wg  (  4) / 0.054904695975835191925936891540473_real64 /
          data wg  (  5) / 0.068038333812356917207187185656708_real64 /
          data wg  (  6) / 0.080140700335001018013234959669111_real64 /
          data wg  (  7) / 0.091028261982963649811497220702892_real64 /
          data wg  (  8) / 0.100535949067050644202206890392686_real64 /
          data wg  (  9) / 0.108519624474263653116093957050117_real64 /
          data wg  ( 10) / 0.114858259145711648339325545869556_real64 /
          data wg  ( 11) / 0.119455763535784772228178126512901_real64 /
          data wg  ( 12) / 0.122242442990310041688959518945852_real64 /
          data wg  ( 13) / 0.123176053726715451203902873079050_real64 /

          data xgk (  1) / 0.999262104992609834193457486540341_real64 /
          data xgk (  2) / 0.995556969790498097908784946893902_real64 /
          data xgk (  3) / 0.988035794534077247637331014577406_real64 /
          data xgk (  4) / 0.976663921459517511498315386479594_real64 /
          data xgk (  5) / 0.961614986425842512418130033660167_real64 /
          data xgk (  6) / 0.942974571228974339414011169658471_real64 /
          data xgk (  7) / 0.920747115281701561746346084546331_real64 /
          data xgk (  8) / 0.894991997878275368851042006782805_real64 /
          data xgk (  9) / 0.865847065293275595448996969588340_real64 /
          data xgk ( 10) / 0.833442628760834001421021108693570_real64 /
          data xgk ( 11) / 0.797873797998500059410410904994307_real64 /
          data xgk ( 12) / 0.759259263037357630577282865204361_real64 /
          data xgk ( 13) / 0.717766406813084388186654079773298_real64 /
          data xgk ( 14) / 0.673566368473468364485120633247622_real64 /
          data xgk ( 15) / 0.626810099010317412788122681624518_real64 /
          data xgk ( 16) / 0.577662930241222967723689841612654_real64 /
          data xgk ( 17) / 0.526325284334719182599623778158010_real64 /
          data xgk ( 18) / 0.473002731445714960522182115009192_real64 /
          data xgk ( 19) / 0.417885382193037748851814394594572_real64 /
          data xgk ( 20) / 0.361172305809387837735821730127641_real64 /
          data xgk ( 21) / 0.303089538931107830167478909980339_real64 /
          data xgk ( 22) / 0.243866883720988432045190362797452_real64 /
          data xgk ( 23) / 0.183718939421048892015969888759528_real64 /
          data xgk ( 24) / 0.122864692610710396387359818808037_real64 /
          data xgk ( 25) / 0.061544483005685078886546392366797_real64 /
          data xgk ( 26) / 0.000000000000000000000000000000000_real64 /

          data wgk (  1) / 0.001987383892330315926507851882843_real64 /
          data wgk (  2) / 0.005561932135356713758040236901066_real64 /
          data wgk (  3) / 0.009473973386174151607207710523655_real64 /
          data wgk (  4) / 0.013236229195571674813656405846976_real64 /
          data wgk (  5) / 0.016847817709128298231516667536336_real64 /
          data wgk (  6) / 0.020435371145882835456568292235939_real64 /
          data wgk (  7) / 0.024009945606953216220092489164881_real64 /
          data wgk (  8) / 0.027475317587851737802948455517811_real64 /
          data wgk (  9) / 0.030792300167387488891109020215229_real64 /
          data wgk ( 10) / 0.034002130274329337836748795229551_real64 /
          data wgk ( 11) / 0.037116271483415543560330625367620_real64 /
          data wgk ( 12) / 0.040083825504032382074839284467076_real64 /
          data wgk ( 13) / 0.042872845020170049476895792439495_real64 /
          data wgk ( 14) / 0.045502913049921788909870584752660_real64 /
          data wgk ( 15) / 0.047982537138836713906392255756915_real64 /
          data wgk ( 16) / 0.050277679080715671963325259433440_real64 /
          data wgk ( 17) / 0.052362885806407475864366712137873_real64 /
          data wgk ( 18) / 0.054251129888545490144543370459876_real64 /
          data wgk ( 19) / 0.055950811220412317308240686382747_real64 /
          data wgk ( 20) / 0.057437116361567832853582693939506_real64 /
          data wgk ( 21) / 0.058689680022394207961974175856788_real64 /
          data wgk ( 22) / 0.059720340324174059979099291932562_real64 /
          data wgk ( 23) / 0.060539455376045862945360267517565_real64 /
          data wgk ( 24) / 0.061128509717053048305859030416293_real64 /
          data wgk ( 25) / 0.061471189871425316661544131965264_real64 /
          data wgk ( 26) / 0.061580818067832935078759824240066_real64 /

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          centr = 0.5_real64*(a+b)
          hlgth = 0.5_real64*(b-a)
          dhlgth =  abs ( hlgth)
          !
          !  compute the 51-point kronrod approximation to
          !  the integral, and estimate the absolute error.
          !
          fc = f(centr)
          resg = wg(13)*fc
          resk = wgk(26)*fc
          resabs =  abs ( resk)

          do j=1,12
             jtw = j*2
             absc = hlgth*xgk(jtw)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtw) = fval1
             fv2(jtw) = fval2
             fsum = fval1+fval2
             resg = resg+wg(j)*fsum
             resk = resk+wgk(jtw)*fsum
             resabs = resabs+wgk(jtw)*( abs ( fval1)+ abs ( fval2))
          end do

          do j = 1,13
             jtwm1 = j*2-1
             absc = hlgth*xgk(jtwm1)
             fval1 = f(centr-absc)
             fval2 = f(centr+absc)
             fv1(jtwm1) = fval1
             fv2(jtwm1) = fval2
             fsum = fval1+fval2
             resk = resk+wgk(jtwm1)*fsum
             resabs = resabs+wgk(jtwm1)*( abs ( fval1)+ abs ( fval2))
          end do

          reskh = resk*0.5_real64
          resasc = wgk(26)* abs ( fc-reskh)

          do j=1,25
             resasc = resasc+wgk(j)*( abs ( fv1(j)-reskh)+ abs ( fv2(j)-reskh))
          end do

          result = resk*hlgth
          resabs = resabs*dhlgth
          resasc = resasc*dhlgth
          abserr =  abs ( (resk-resg)*hlgth)
          if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) &
               abserr = resasc* min (0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
          if(resabs.gt.uflow/(0.5e2_real64*epmach)) abserr = max &
               ((epmach*0.5e2_real64)*resabs,abserr)

          return
        end subroutine dqk51
        subroutine dqk61(f,a,b,result,abserr,resabs,resasc)

          !*****************************************************************************80
          !
          !! DQK61 carries out a 61 point Gauss-Kronrod quadrature rule.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  to compute i = integral of f over (a,b) with error
          !                     estimate
          !                 j = integral of  abs ( f) over (a,b)
          !
          !  Parameters:
          !
          !   on entry
          !     f      - real(real64)
          !              function subprogram defining the integrand
          !              function f(x). the actual name for f needs to be
          !              declared e x t e r n a l in the calling program.
          !
          !     a      - real(real64)
          !              lower limit of integration
          !
          !     b      - real(real64)
          !              upper limit of integration
          !
          !   on return
          !     result - real(real64)
          !              approximation to the integral i
          !              result is computed by applying the 61-point
          !              kronrod rule (resk) obtained by optimal addition of
          !              abscissae to the 30-point gauss rule (resg).
          !
          !     abserr - real(real64)
          !              estimate of the modulus of the absolute error,
          !              which should equal or exceed  abs ( i-result)
          !
          !     resabs - real(real64)
          !              approximation to the integral j
          !
          !     resasc - real(real64)
          !              approximation to the integral of  abs ( f-i/(b-a))
          !
          !  Local Parameters:
          !
          !     the abscissae and weights are given for the
          !     interval (-1,1). because of symmetry only the positive
          !     abscissae and their corresponding weights are given.
          !
          !     xgk   - abscissae of the 61-point kronrod rule
          !             xgk(2), xgk(4)  ... abscissae of the 30-point
          !             gauss rule
          !             xgk(1), xgk(3)  ... optimally added abscissae
          !             to the 30-point gauss rule
          !
          !     wgk   - weights of the 61-point kronrod rule
          !
          !     wg    - weigths of the 30-point gauss rule
          !
          !
          !   gauss quadrature weights and kronron quadrature abscissae and weights
          !   as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
          !   bell labs, nov. 1981.
          !
          !     centr  - mid point of the interval
          !     hlgth  - half-length of the interval
          !     dabsc  - abscissa
          !     fval*  - function value
          !     resg   - result of the 30-point gauss rule
          !     resk   - result of the 61-point kronrod rule
          !     reskh  - approximation to the mean value of f
          !              over (a,b), i.e. to i/(b-a)
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,dabsc,abserr,b,centr,dhlgth, &
               epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc, &
               resg,resk,reskh,result,uflow,wg,wgk,xgk
          integer(int32) j,jtw,jtwm1
          external f

          dimension fv1(30),fv2(30),xgk(31),wgk(31),wg(15)

          data wg  (  1) / 0.007968192496166605615465883474674_real64 /
          data wg  (  2) / 0.018466468311090959142302131912047_real64 /
          data wg  (  3) / 0.028784707883323369349719179611292_real64 /
          data wg  (  4) / 0.038799192569627049596801936446348_real64 /
          data wg  (  5) / 0.048402672830594052902938140422808_real64 /
          data wg  (  6) / 0.057493156217619066481721689402056_real64 /
          data wg  (  7) / 0.065974229882180495128128515115962_real64 /
          data wg  (  8) / 0.073755974737705206268243850022191_real64 /
          data wg  (  9) / 0.080755895229420215354694938460530_real64 /
          data wg  ( 10) / 0.086899787201082979802387530715126_real64 /
          data wg  ( 11) / 0.092122522237786128717632707087619_real64 /
          data wg  ( 12) / 0.096368737174644259639468626351810_real64 /
          data wg  ( 13) / 0.099593420586795267062780282103569_real64 /
          data wg  ( 14) / 0.101762389748405504596428952168554_real64 /
          data wg  ( 15) / 0.102852652893558840341285636705415_real64 /

          data xgk (  1) / 0.999484410050490637571325895705811_real64 /
          data xgk (  2) / 0.996893484074649540271630050918695_real64 /
          data xgk (  3) / 0.991630996870404594858628366109486_real64 /
          data xgk (  4) / 0.983668123279747209970032581605663_real64 /
          data xgk (  5) / 0.973116322501126268374693868423707_real64 /
          data xgk (  6) / 0.960021864968307512216871025581798_real64 /
          data xgk (  7) / 0.944374444748559979415831324037439_real64 /
          data xgk (  8) / 0.926200047429274325879324277080474_real64 /
          data xgk (  9) / 0.905573307699907798546522558925958_real64 /
          data xgk ( 10) / 0.882560535792052681543116462530226_real64 /
          data xgk ( 11) / 0.857205233546061098958658510658944_real64 /
          data xgk ( 12) / 0.829565762382768397442898119732502_real64 /
          data xgk ( 13) / 0.799727835821839083013668942322683_real64 /
          data xgk ( 14) / 0.767777432104826194917977340974503_real64 /
          data xgk ( 15) / 0.733790062453226804726171131369528_real64 /
          data xgk ( 16) / 0.697850494793315796932292388026640_real64 /
          data xgk ( 17) / 0.660061064126626961370053668149271_real64 /
          data xgk ( 18) / 0.620526182989242861140477556431189_real64 /
          data xgk ( 19) / 0.579345235826361691756024932172540_real64 /
          data xgk ( 20) / 0.536624148142019899264169793311073_real64 /
          data xgk ( 21) / 0.492480467861778574993693061207709_real64 /
          data xgk ( 22) / 0.447033769538089176780609900322854_real64 /
          data xgk ( 23) / 0.400401254830394392535476211542661_real64 /
          data xgk ( 24) / 0.352704725530878113471037207089374_real64 /
          data xgk ( 25) / 0.304073202273625077372677107199257_real64 /
          data xgk ( 26) / 0.254636926167889846439805129817805_real64 /
          data xgk ( 27) / 0.204525116682309891438957671002025_real64 /
          data xgk ( 28) / 0.153869913608583546963794672743256_real64 /
          data xgk ( 29) / 0.102806937966737030147096751318001_real64 /
          data xgk ( 30) / 0.051471842555317695833025213166723_real64 /
          data xgk ( 31) / 0.000000000000000000000000000000000_real64 /

          data wgk (  1) / 0.001389013698677007624551591226760_real64 /
          data wgk (  2) / 0.003890461127099884051267201844516_real64 /
          data wgk (  3) / 0.006630703915931292173319826369750_real64 /
          data wgk (  4) / 0.009273279659517763428441146892024_real64 /
          data wgk (  5) / 0.011823015253496341742232898853251_real64 /
          data wgk (  6) / 0.014369729507045804812451432443580_real64 /
          data wgk (  7) / 0.016920889189053272627572289420322_real64 /
          data wgk (  8) / 0.019414141193942381173408951050128_real64 /
          data wgk (  9) / 0.021828035821609192297167485738339_real64 /
          data wgk ( 10) / 0.024191162078080601365686370725232_real64 /
          data wgk ( 11) / 0.026509954882333101610601709335075_real64 /
          data wgk ( 12) / 0.028754048765041292843978785354334_real64 /
          data wgk ( 13) / 0.030907257562387762472884252943092_real64 /
          data wgk ( 14) / 0.032981447057483726031814191016854_real64 /
          data wgk ( 15) / 0.034979338028060024137499670731468_real64 /
          data wgk ( 16) / 0.036882364651821229223911065617136_real64 /
          data wgk ( 17) / 0.038678945624727592950348651532281_real64 /
          data wgk ( 18) / 0.040374538951535959111995279752468_real64 /
          data wgk ( 19) / 0.041969810215164246147147541285970_real64 /
          data wgk ( 20) / 0.043452539701356069316831728117073_real64 /
          data wgk ( 21) / 0.044814800133162663192355551616723_real64 /
          data wgk ( 22) / 0.046059238271006988116271735559374_real64 /
          data wgk ( 23) / 0.047185546569299153945261478181099_real64 /
          data wgk ( 24) / 0.048185861757087129140779492298305_real64 /
          data wgk ( 25) / 0.049055434555029778887528165367238_real64 /
          data wgk ( 26) / 0.049795683427074206357811569379942_real64 /
          data wgk ( 27) / 0.050405921402782346840893085653585_real64 /
          data wgk ( 28) / 0.050881795898749606492297473049805_real64 /
          data wgk ( 29) / 0.051221547849258772170656282604944_real64 /
          data wgk ( 30) / 0.051426128537459025933862879215781_real64 /
          data wgk ( 31) / 0.051494729429451567558340433647099_real64 /

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          centr = 0.5_real64*(b+a)
          hlgth = 0.5_real64*(b-a)
          dhlgth =  abs ( hlgth)
          !
          !  compute the 61-point kronrod approximation to the
          !  integral, and estimate the absolute error.
          !
          resg = 0.0_real64
          fc = f(centr)
          resk = wgk(31)*fc
          resabs =  abs ( resk)

          do j=1,15
             jtw = j*2
             dabsc = hlgth*xgk(jtw)
             fval1 = f(centr-dabsc)
             fval2 = f(centr+dabsc)
             fv1(jtw) = fval1
             fv2(jtw) = fval2
             fsum = fval1+fval2
             resg = resg+wg(j)*fsum
             resk = resk+wgk(jtw)*fsum
             resabs = resabs+wgk(jtw)*( abs ( fval1)+ abs ( fval2))
          end do

          do j=1,15
             jtwm1 = j*2-1
             dabsc = hlgth*xgk(jtwm1)
             fval1 = f(centr-dabsc)
             fval2 = f(centr+dabsc)
             fv1(jtwm1) = fval1
             fv2(jtwm1) = fval2
             fsum = fval1+fval2
             resk = resk+wgk(jtwm1)*fsum
             resabs = resabs+wgk(jtwm1)*( abs ( fval1)+ abs ( fval2))
          end do

          reskh = resk*0.5_real64
          resasc = wgk(31)* abs ( fc-reskh)

          do j=1,30
             resasc = resasc+wgk(j)*( abs ( fv1(j)-reskh)+ abs ( fv2(j)-reskh))
          end do

          result = resk*hlgth
          resabs = resabs*dhlgth
          resasc = resasc*dhlgth
          abserr =  abs ( (resk-resg)*hlgth)
          if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) &
               abserr = resasc* min (0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
          if(resabs.gt.uflow/(0.5e2_real64*epmach)) abserr = max &
               ((epmach*0.5e2_real64)*resabs,abserr)

          return
        end subroutine dqk61
        subroutine dqmomo(alfa,beta,ri,rj,rg,rh,integr)

          !*****************************************************************************80
          !
          !! DQMOMO computes modified Chebyshev moments.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  this routine computes modified chebsyshev moments. the k-th
          !      modified chebyshev moment is defined as the integral over
          !      (-1,1) of w(x)*t(k,x), where t(k,x) is the chebyshev
          !      polynomial of degree k.
          !
          !  Parameters:
          !
          !     alfa   - real(real64)
          !              parameter in the weight function w(x), alfa.gt.(-1)
          !
          !     beta   - real(real64)
          !              parameter in the weight function w(x), beta.gt.(-1)
          !
          !     ri     - real(real64)
          !              vector of dimension 25
          !              ri(k) is the integral over (-1,1) of
          !              (1+x)**alfa*t(k-1,x), k = 1, ..., 25.
          !
          !     rj     - real(real64)
          !              vector of dimension 25
          !              rj(k) is the integral over (-1,1) of
          !              (1-x)**beta*t(k-1,x), k = 1, ..., 25.
          !
          !     rg     - real(real64)
          !              vector of dimension 25
          !              rg(k) is the integral over (-1,1) of
          !              (1+x)**alfa*log((1+x)/2)*t(k-1,x), k = 1, ..., 25.
          !
          !     rh     - real(real64)
          !              vector of dimension 25
          !              rh(k) is the integral over (-1,1) of
          !              (1-x)**beta*log((1-x)/2)*t(k-1,x), k = 1, ..., 25.
          !
          !     integr - integer(int32)
          !              input parameter indicating the modified
          !              moments to be computed
          !              integr = 1 compute ri, rj
          !                     = 2 compute ri, rj, rg
          !                     = 3 compute ri, rj, rh
          !                     = 4 compute ri, rj, rg, rh
          !
          implicit none

          real(real64) alfa,alfp1,alfp2,an,anm1,beta,betp1,betp2,ralf, &
               rbet,rg,rh,ri,rj
          integer(int32) i,im1,integr
          dimension rg(25),rh(25),ri(25),rj(25)

          alfp1 = alfa+0.1e1_real64
          betp1 = beta+0.1e1_real64
          alfp2 = alfa+0.2e1_real64
          betp2 = beta+0.2e1_real64
          ralf = 0.2e1_real64**alfp1
          rbet = 0.2e1_real64**betp1
          !
          !  compute ri, rj using a forward recurrence relation.
          !
          ri(1) = ralf/alfp1
          rj(1) = rbet/betp1
          ri(2) = ri(1)*alfa/alfp2
          rj(2) = rj(1)*beta/betp2
          an = 0.2e1_real64
          anm1 = 0.1e1_real64

          do i=3,25
             ri(i) = -(ralf+an*(an-alfp2)*ri(i-1))/(anm1*(an+alfp1))
             rj(i) = -(rbet+an*(an-betp2)*rj(i-1))/(anm1*(an+betp1))
             anm1 = an
             an = an+0.1e1_real64
          end do

          if(integr.eq.1) go to 70
          if(integr.eq.3) go to 40
          !
          !  compute rg using a forward recurrence relation.
          !
          rg(1) = -ri(1)/alfp1
          rg(2) = -(ralf+ralf)/(alfp2*alfp2)-rg(1)
          an = 0.2e1_real64
          anm1 = 0.1e1_real64
          im1 = 2

          do i=3,25
             rg(i) = -(an*(an-alfp2)*rg(im1)-an*ri(im1)+anm1*ri(i))/ &
                  (anm1*(an+alfp1))
             anm1 = an
             an = an+0.1e1_real64
             im1 = i
          end do

          if(integr.eq.2) go to 70
          !
          !  compute rh using a forward recurrence relation.
          !
40        rh(1) = -rj(1)/betp1
          rh(2) = -(rbet+rbet)/(betp2*betp2)-rh(1)
          an = 0.2e1_real64
          anm1 = 0.1e1_real64
          im1 = 2

          do i=3,25
             rh(i) = -(an*(an-betp2)*rh(im1)-an*rj(im1)+ &
                  anm1*rj(i))/(anm1*(an+betp1))
             anm1 = an
             an = an+0.1e1_real64
             im1 = i
          end do

          do i=2,25,2
             rh(i) = -rh(i)
          end do

70        continue

          do i=2,25,2
             rj(i) = -rj(i)
          end do

90        continue

          return
        end subroutine dqmomo
        subroutine dqng ( f, a, b, epsabs, epsrel, result, abserr, neval, ier )

          !*****************************************************************************80
          !
          !! DQNG estimates an integral, using non-adaptive integration.
          !
          !  Modified:
          !
          !    11 September 2015
          !
          !  Author:
          !
          !    Robert Piessens, Elise de Doncker
          !
          !***purpose  the routine calculates an approximation result to a
          !      given definite integral i = integral of f over (a,b),
          !      hopefully satisfying following claim for accuracy
          !      abs(i-result).le.max(epsabs,epsrel*abs(i)).
          !
          !  Parameters:
          !
          !     f      - real(real64)
          !              function subprogram defining the integrand function
          !              f(x). the actual name for f needs to be declared
          !              e x t e r n a l in the driver program.
          !
          !     a      - real(real64)
          !              lower limit of integration
          !
          !     b      - real(real64)
          !              upper limit of integration
          !
          !     epsabs - real(real64)
          !              absolute accuracy requested
          !     epsrel - real(real64)
          !              relative accuracy requested
          !              if  epsabs.le.0
          !              and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
          !              the routine will end with ier = 6.
          !
          !   on return
          !     result - real(real64)
          !              approximation to the integral i
          !              result is obtained by applying the 21-point
          !              gauss-kronrod rule (res21) obtained by optimal
          !              addition of abscissae to the 10-point gauss rule
          !              (res10), or by applying the 43-point rule (res43)
          !              obtained by optimal addition of abscissae to the
          !              21-point gauss-kronrod rule, or by applying the
          !              87-point rule (res87) obtained by optimal addition
          !              of abscissae to the 43-point rule.
          !
          !     abserr - real(real64)
          !              estimate of the modulus of the absolute error,
          !              which should equal or exceed abs(i-result)
          !
          !     neval  - integer(int32)
          !              number of integrand evaluations
          !
          !     ier    - ier = 0 normal and reliable termination of the
          !                      routine. it is assumed that the requested
          !                      accuracy has been achieved.
          !              ier.gt.0 abnormal termination of the routine. it is
          !                      assumed that the requested accuracy has
          !                      not been achieved.
          !     error messages
          !              ier = 1 the maximum number of steps has been
          !                      executed. the integral is probably too
          !                      difficult to be calculated by dqng.
          !                  = 6 the input is invalid, because
          !                      epsabs.le.0 and
          !                      epsrel.lt.max(50*rel.mach.acc.,0.5d-28).
          !                      result, abserr and neval are set to zero.
          !
          !  Local Parameters:
          !
          !     the data statements contain the
          !     abscissae and weights of the integration rules used.
          !
          !     x1      abscissae common to the 10-, 21-, 43- and 87-
          !             point rule
          !     x2      abscissae common to the 21-, 43- and 87-point rule
          !     x3      abscissae common to the 43- and 87-point rule
          !     x4      abscissae of the 87-point rule
          !     w10     weights of the 10-point formula
          !     w21a    weights of the 21-point formula for abscissae x1
          !     w21b    weights of the 21-point formula for abscissae x2
          !     w43a    weights of the 43-point formula for abscissae x1, x3
          !     w43b    weights of the 43-point formula for abscissae x3
          !     w87a    weights of the 87-point formula for abscissae x1,
          !             x2, x3
          !     w87b    weights of the 87-point formula for abscissae x4
          !
          !
          ! gauss-kronrod-patterson quadrature coefficients for use in
          ! quadpack routine qng.  these coefficients were calculated with
          ! 101 decimal digit arithmetic by l. w. fullerton, bell labs, nov 1981.
          !
          !     centr  - mid point of the integration interval
          !     hlgth  - half-length of the integration interval
          !     fcentr - function value at mid point
          !     absc   - abscissa
          !     fval   - function value
          !     savfun - array of function values which have already been
          !              computed
          !     res10  - 10-point gauss result
          !     res21  - 21-point kronrod result
          !     res43  - 43-point result
          !     res87  - 87-point result
          !     resabs - approximation to the integral of abs(f)
          !     resasc - approximation to the integral of abs(f-i/(b-a))
          !
          !     machine dependent constants
          !
          !     epmach is the largest relative spacing.
          !     uflow is the smallest positive magnitude.
          !
          implicit none

          real(real64) a,absc,abserr,b,centr,dhlgth, &
               epmach,epsabs,epsrel,f,fcentr,fval,fval1,fval2,fv1,fv2, &
               fv3,fv4,hlgth,result,res10,res21,res43,res87,resabs,resasc, &
               reskh,savfun,uflow,w10,w21a,w21b,w43a,w43b,w87a,w87b,x1,x2,x3,x4
          integer(int32) ier,ipx,k,l,neval
          external f
          dimension fv1(5),fv2(5),fv3(5),fv4(5),x1(5),x2(5),x3(11),x4(22), &
               w10(5),w21a(5),w21b(6),w43a(10),w43b(12),w87a(21),w87b(23), &
               savfun(21)

          data x1    (  1) / 0.973906528517171720077964012084452_real64 /
          data x1    (  2) / 0.865063366688984510732096688423493_real64 /
          data x1    (  3) / 0.679409568299024406234327365114874_real64 /
          data x1    (  4) / 0.433395394129247190799265943165784_real64 /
          data x1    (  5) / 0.148874338981631210884826001129720_real64 /
          data w10   (  1) / 0.066671344308688137593568809893332_real64 /
          data w10   (  2) / 0.149451349150580593145776339657697_real64 /
          data w10   (  3) / 0.219086362515982043995534934228163_real64 /
          data w10   (  4) / 0.269266719309996355091226921569469_real64 /
          data w10   (  5) / 0.295524224714752870173892994651338_real64 /

          data x2    (  1) / 0.995657163025808080735527280689003_real64 /
          data x2    (  2) / 0.930157491355708226001207180059508_real64 /
          data x2    (  3) / 0.780817726586416897063717578345042_real64 /
          data x2    (  4) / 0.562757134668604683339000099272694_real64 /
          data x2    (  5) / 0.294392862701460198131126603103866_real64 /
          data w21a  (  1) / 0.032558162307964727478818972459390_real64 /
          data w21a  (  2) / 0.075039674810919952767043140916190_real64 /
          data w21a  (  3) / 0.109387158802297641899210590325805_real64 /
          data w21a  (  4) / 0.134709217311473325928054001771707_real64 /
          data w21a  (  5) / 0.147739104901338491374841515972068_real64 /
          data w21b  (  1) / 0.011694638867371874278064396062192_real64 /
          data w21b  (  2) / 0.054755896574351996031381300244580_real64 /
          data w21b  (  3) / 0.093125454583697605535065465083366_real64 /
          data w21b  (  4) / 0.123491976262065851077958109831074_real64 /
          data w21b  (  5) / 0.142775938577060080797094273138717_real64 /
          data w21b  (  6) / 0.149445554002916905664936468389821_real64 /
          !
          data x3    (  1) / 0.999333360901932081394099323919911_real64 /
          data x3    (  2) / 0.987433402908088869795961478381209_real64 /
          data x3    (  3) / 0.954807934814266299257919200290473_real64 /
          data x3    (  4) / 0.900148695748328293625099494069092_real64 /
          data x3    (  5) / 0.825198314983114150847066732588520_real64 /
          data x3    (  6) / 0.732148388989304982612354848755461_real64 /
          data x3    (  7) / 0.622847970537725238641159120344323_real64 /
          data x3    (  8) / 0.499479574071056499952214885499755_real64 /
          data x3    (  9) / 0.364901661346580768043989548502644_real64 /
          data x3    ( 10) / 0.222254919776601296498260928066212_real64 /
          data x3    ( 11) / 0.074650617461383322043914435796506_real64 /
          data w43a  (  1) / 0.016296734289666564924281974617663_real64 /
          data w43a  (  2) / 0.037522876120869501461613795898115_real64 /
          data w43a  (  3) / 0.054694902058255442147212685465005_real64 /
          data w43a  (  4) / 0.067355414609478086075553166302174_real64 /
          data w43a  (  5) / 0.073870199632393953432140695251367_real64 /
          data w43a  (  6) / 0.005768556059769796184184327908655_real64 /
          data w43a  (  7) / 0.027371890593248842081276069289151_real64 /
          data w43a  (  8) / 0.046560826910428830743339154433824_real64 /
          data w43a  (  9) / 0.061744995201442564496240336030883_real64 /
          data w43a  ( 10) / 0.071387267268693397768559114425516_real64 /
          data w43b  (  1) / 0.001844477640212414100389106552965_real64 /
          data w43b  (  2) / 0.010798689585891651740465406741293_real64 /
          data w43b  (  3) / 0.021895363867795428102523123075149_real64 /
          data w43b  (  4) / 0.032597463975345689443882222526137_real64 /
          data w43b  (  5) / 0.042163137935191811847627924327955_real64 /
          data w43b  (  6) / 0.050741939600184577780189020092084_real64 /
          data w43b  (  7) / 0.058379395542619248375475369330206_real64 /
          data w43b  (  8) / 0.064746404951445885544689259517511_real64 /
          data w43b  (  9) / 0.069566197912356484528633315038405_real64 /
          data w43b  ( 10) / 0.072824441471833208150939535192842_real64 /
          data w43b  ( 11) / 0.074507751014175118273571813842889_real64 /
          data w43b  ( 12) / 0.074722147517403005594425168280423_real64 /

          data x4    (  1) / 0.999902977262729234490529830591582_real64 /
          data x4    (  2) / 0.997989895986678745427496322365960_real64 /
          data x4    (  3) / 0.992175497860687222808523352251425_real64 /
          data x4    (  4) / 0.981358163572712773571916941623894_real64 /
          data x4    (  5) / 0.965057623858384619128284110607926_real64 /
          data x4    (  6) / 0.943167613133670596816416634507426_real64 /
          data x4    (  7) / 0.915806414685507209591826430720050_real64 /
          data x4    (  8) / 0.883221657771316501372117548744163_real64 /
          data x4    (  9) / 0.845710748462415666605902011504855_real64 /
          data x4    ( 10) / 0.803557658035230982788739474980964_real64 /
          data x4    ( 11) / 0.757005730685495558328942793432020_real64 /
          data x4    ( 12) / 0.706273209787321819824094274740840_real64 /
          data x4    ( 13) / 0.651589466501177922534422205016736_real64 /
          data x4    ( 14) / 0.593223374057961088875273770349144_real64 /
          data x4    ( 15) / 0.531493605970831932285268948562671_real64 /
          data x4    ( 16) / 0.466763623042022844871966781659270_real64 /
          data x4    ( 17) / 0.399424847859218804732101665817923_real64 /
          data x4    ( 18) / 0.329874877106188288265053371824597_real64 /
          data x4    ( 19) / 0.258503559202161551802280975429025_real64 /
          data x4    ( 20) / 0.185695396568346652015917141167606_real64 /
          data x4    ( 21) / 0.111842213179907468172398359241362_real64 /
          data x4    ( 22) / 0.037352123394619870814998165437704_real64 /
          data w87a  (  1) / 0.008148377384149172900002878448190_real64 /
          data w87a  (  2) / 0.018761438201562822243935059003794_real64 /
          data w87a  (  3) / 0.027347451050052286161582829741283_real64 /
          data w87a  (  4) / 0.033677707311637930046581056957588_real64 /
          data w87a  (  5) / 0.036935099820427907614589586742499_real64 /
          data w87a  (  6) / 0.002884872430211530501334156248695_real64 /
          data w87a  (  7) / 0.013685946022712701888950035273128_real64 /
          data w87a  (  8) / 0.023280413502888311123409291030404_real64 /
          data w87a  (  9) / 0.030872497611713358675466394126442_real64 /
          data w87a  ( 10) / 0.035693633639418770719351355457044_real64 /
          data w87a  ( 11) / 0.000915283345202241360843392549948_real64 /
          data w87a  ( 12) / 0.005399280219300471367738743391053_real64 /
          data w87a  ( 13) / 0.010947679601118931134327826856808_real64 /
          data w87a  ( 14) / 0.016298731696787335262665703223280_real64 /
          data w87a  ( 15) / 0.021081568889203835112433060188190_real64 /
          data w87a  ( 16) / 0.025370969769253827243467999831710_real64 /
          data w87a  ( 17) / 0.029189697756475752501446154084920_real64 /
          data w87a  ( 18) / 0.032373202467202789685788194889595_real64 /
          data w87a  ( 19) / 0.034783098950365142750781997949596_real64 /
          data w87a  ( 20) / 0.036412220731351787562801163687577_real64 /
          data w87a  ( 21) / 0.037253875503047708539592001191226_real64 /
          data w87b  (  1) / 0.000274145563762072350016527092881_real64 /
          data w87b  (  2) / 0.001807124155057942948341311753254_real64 /
          data w87b  (  3) / 0.004096869282759164864458070683480_real64 /
          data w87b  (  4) / 0.006758290051847378699816577897424_real64 /
          data w87b  (  5) / 0.009549957672201646536053581325377_real64 /
          data w87b  (  6) / 0.012329447652244853694626639963780_real64 /
          data w87b  (  7) / 0.015010447346388952376697286041943_real64 /
          data w87b  (  8) / 0.017548967986243191099665352925900_real64 /
          data w87b  (  9) / 0.019938037786440888202278192730714_real64 /
          data w87b  ( 10) / 0.022194935961012286796332102959499_real64 /
          data w87b  ( 11) / 0.024339147126000805470360647041454_real64 /
          data w87b  ( 12) / 0.026374505414839207241503786552615_real64 /
          data w87b  ( 13) / 0.028286910788771200659968002987960_real64 /
          data w87b  ( 14) / 0.030052581128092695322521110347341_real64 /
          data w87b  ( 15) / 0.031646751371439929404586051078883_real64 /
          data w87b  ( 16) / 0.033050413419978503290785944862689_real64 /
          data w87b  ( 17) / 0.034255099704226061787082821046821_real64 /
          data w87b  ( 18) / 0.035262412660156681033782717998428_real64 /
          data w87b  ( 19) / 0.036076989622888701185500318003895_real64 /
          data w87b  ( 20) / 0.036698604498456094498018047441094_real64 /
          data w87b  ( 21) / 0.037120549269832576114119958413599_real64 /
          data w87b  ( 22) / 0.037334228751935040321235449094698_real64 /
          data w87b  ( 23) / 0.037361073762679023410321241766599_real64 /

          epmach = epsilon ( epmach )
          uflow = tiny ( uflow )
          !
          !  test on validity of parameters
          !
          result = 0.0_real64
          abserr = 0.0_real64
          neval = 0
          ier = 6
          if(epsabs.le.0.0_real64.and.epsrel.lt. max ( 0.5e2_real64*epmach,0.5e-28_real64)) &
               go to 80
          hlgth = 0.5_real64*(b-a)
          dhlgth =  abs ( hlgth)
          centr = 0.5_real64*(b+a)
          fcentr = f(centr)
          neval = 21
          ier = 1
          !
          !  compute the integral using the 10- and 21-point formula.
          !
          do 70 l = 1,3

             go to (5,25,45),l

5            res10 = 0.0_real64
             res21 = w21b(6)*fcentr
             resabs = w21b(6)* abs ( fcentr)

             do k=1,5
                absc = hlgth*x1(k)
                fval1 = f(centr+absc)
                fval2 = f(centr-absc)
                fval = fval1+fval2
                res10 = res10+w10(k)*fval
                res21 = res21+w21a(k)*fval
                resabs = resabs+w21a(k)*( abs ( fval1)+ abs ( fval2))
                savfun(k) = fval
                fv1(k) = fval1
                fv2(k) = fval2
             end do

             ipx = 5

             do k=1,5
                ipx = ipx+1
                absc = hlgth*x2(k)
                fval1 = f(centr+absc)
                fval2 = f(centr-absc)
                fval = fval1+fval2
                res21 = res21+w21b(k)*fval
                resabs = resabs+w21b(k)*( abs ( fval1)+ abs ( fval2))
                savfun(ipx) = fval
                fv3(k) = fval1
                fv4(k) = fval2
             end do
             !
             !  test for convergence.
             !
             result = res21*hlgth
             resabs = resabs*dhlgth
             reskh = 0.5_real64*res21
             resasc = w21b(6)* abs ( fcentr-reskh)

             do k = 1,5
                resasc = resasc+w21a(k)*( abs ( fv1(k)-reskh)+ abs ( fv2(k)-reskh)) &
                     +w21b(k)*( abs ( fv3(k)-reskh)+ abs ( fv4(k)-reskh))
             end do

             abserr =  abs ( (res21-res10)*hlgth)
             resasc = resasc*dhlgth
             go to 65
             !
             !  compute the integral using the 43-point formula.
             !
25           res43 = w43b(12)*fcentr
             neval = 43

             do k=1,10
                res43 = res43+savfun(k)*w43a(k)
             end do

             do k=1,11
                ipx = ipx+1
                absc = hlgth*x3(k)
                fval = f(absc+centr)+f(centr-absc)
                res43 = res43+fval*w43b(k)
                savfun(ipx) = fval
             end do
             !
             !  test for convergence.
             !
             result = res43*hlgth
             abserr =  abs ( (res43-res21)*hlgth)
             go to 65
             !
             !  compute the integral using the 87-point formula.
             !
45           res87 = w87b(23)*fcentr
             neval = 87

             do k=1,21
                res87 = res87+savfun(k)*w87a(k)
             end do

             do k=1,22
                absc = hlgth*x4(k)
                res87 = res87+w87b(k)*(f(absc+centr)+f(centr-absc))
             end do

             result = res87*hlgth
             abserr =  abs ( (res87-res43)*hlgth)

65           continue

             if(resasc.ne.0.0_real64.and.abserr.ne.0.0_real64) then
                abserr = resasc* min (0.1e1_real64,(0.2e3_real64*abserr/resasc)**1.5_real64)
             end if

             if (resabs.gt.uflow/(0.5e2_real64*epmach)) then
                abserr = max ((epmach*0.5e2_real64)*resabs,abserr)
             end if

             if (abserr.le. max ( epsabs,epsrel* abs ( result))) then
                ier = 0
                return
             end if

70           continue

80           call xerror('abnormal return from dqng ',26,ier,0)
999          continue

             return
           end subroutine dqng
           subroutine dqpsrt ( limit, last, maxerr, ermax, elist, iord, nrmax )

             !*****************************************************************************80
             !
             !! DQPSRT maintains the order of a list of local error estimates.
             !
             !  Modified:
             !
             !    11 September 2015
             !
             !  Author:
             !
             !    Robert Piessens, Elise de Doncker
             !
             !***purpose  this routine maintains the descending ordering in the
             !      list of the local error estimated resulting from the
             !      interval subdivision process. at each call two error
             !      estimates are inserted using the sequential search
             !      method, top-down for the largest error estimate and
             !      bottom-up for the smallest error estimate.
             !
             !  Parameters:
             !
             !        limit  - integer(int32)
             !                 maximum number of error estimates the list
             !                 can contain
             !
             !        last   - integer(int32)
             !                 number of error estimates currently in the list
             !
             !        maxerr - integer(int32)
             !                 maxerr points to the nrmax-th largest error
             !                 estimate currently in the list
             !
             !        ermax  - real(real64)
             !                 nrmax-th largest error estimate
             !                 ermax = elist(maxerr)
             !
             !        elist  - real(real64)
             !                 vector of dimension last containing
             !                 the error estimates
             !
             !        iord   - integer(int32)
             !                 vector of dimension last, the first k elements
             !                 of which contain pointers to the error
             !                 estimates, such that
             !                 elist(iord(1)),...,  elist(iord(k))
             !                 form a decreasing sequence, with
             !                 k = last if last.le.(limit/2+2), and
             !                 k = limit+1-last otherwise
             !
             !        nrmax  - integer(int32)
             !                 maxerr = iord(nrmax)
             !
             implicit none

             real(real64) elist,ermax,errmax,errmin
             integer(int32) i,ibeg,ido,iord,isucc,j,jbnd,jupbn,k,last, &
                  lim
             integer(int32) limit
             integer(int32) maxerr
             integer(int32) nrmax
             dimension elist(last),iord(last)
             !
             !  check whether the list contains more than
             !  two error estimates.
             !
             if(last.gt.2) go to 10
             iord(1) = 1
             iord(2) = 2
             go to 90
             !
             !  this part of the routine is only executed if, due to a
             !  difficult integrand, subdivision increased the error
             !  estimate. in the normal case the insert procedure should
             !  start after the nrmax-th largest error estimate.
             !
10           errmax = elist(maxerr)

             ido = nrmax-1
             do i = 1,ido
                isucc = iord(nrmax-1)
                if(errmax.le.elist(isucc)) go to 30
                iord(nrmax) = isucc
                nrmax = nrmax-1
             end do
             !
             !  compute the number of elements in the list to be maintained
             !  in descending order. this number depends on the number of
             !  subdivisions still allowed.
             !
30           jupbn = last
             if(last.gt.(limit/2+2)) jupbn = limit+3-last
             errmin = elist(last)
             !
             !  insert errmax by traversing the list top-down,
             !  starting comparison from the element elist(iord(nrmax+1)).
             !
             jbnd = jupbn-1
             ibeg = nrmax+1

             do i=ibeg,jbnd
                isucc = iord(i)
                if(errmax.ge.elist(isucc)) go to 60
                iord(i-1) = isucc
             end do

             iord(jbnd) = maxerr
             iord(jupbn) = last
             go to 90
             !
             !  insert errmin by traversing the list bottom-up.
             !
60           iord(i-1) = maxerr
             k = jbnd

             do j=i,jbnd
                isucc = iord(k)
                if(errmin.lt.elist(isucc)) go to 80
                iord(k+1) = isucc
                k = k-1
             end do

             iord(i) = last
             go to 90
80           iord(k+1) = last
             !
             !     set maxerr and ermax.
             !
90           maxerr = iord(nrmax)
             ermax = elist(maxerr)

             return
           end subroutine dqpsrt
           function dqwgtc ( x, c, p2, p3, p4, kp )

             !*****************************************************************************80
             !
             !! DQWGTC defines the weight function used by DQC25C.
             !
             !  Modified:
             !
             !    11 September 2015
             !
             !  Author:
             !
             !    Robert Piessens, Elise de Doncker
             !
             implicit none

             real(real64) dqwgtc
             real(real64) c,p2,p3,p4,x
             integer(int32) kp

             dqwgtc = 0.1e1_real64 / ( x - c )

             return
           end function dqwgtc

           subroutine xerror ( xmess, nmess, nerr, level )

             !*****************************************************************************80
             !
             !! XERROR replaces the SLATEC XERROR routine.
             !
             !  Modified:
             !
             !    12 September 2015
             !
             implicit none

             integer(int32) level
             integer(int32) nerr
             integer(int32) nmess
             character ( len = * ) xmess

             if ( 1 <= LEVEL ) then
                WRITE ( *,'(1X,A)') XMESS(1:NMESS)
                WRITE ( *,'('' ERROR NUMBER = '',I5,'', MESSAGE LEVEL = '',I5)') &
                     NERR,LEVEL
             end if

             return
           end subroutine xerror

         end module quadrature
