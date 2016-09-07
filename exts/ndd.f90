! Copyright (C) 2016, Simone Marsili 
! All rights reserved.
! License: BSD 3 clause
! Version: 0.0.1
! Author: Simone Marsili (simomarsili@gmail.com)

module constants
  use iso_fortran_env
  implicit none

  real(real64), parameter :: one = 1.0_real64

end module constants

module dirichlet_mod
  use iso_fortran_env
  implicit none

  integer(int32)              :: alphabet_size
  integer(int32)              :: ndata
  integer(int32)              :: ngtz
  integer(int32), allocatable :: fgtz(:)
  integer(int32), allocatable :: multi_gtz(:)

contains

  subroutine dirichlet_initialize(nc,hist)

    integer(int32), intent(in) :: nc
    integer(int32), intent(in) :: hist(:)
    integer(int32)              :: nbins
    integer(int32)              :: i,k,ni
    integer(int32)              :: err
    integer(int32)              :: nmax
    integer(int32), allocatable :: multiplicities(:)

    alphabet_size = nc 
    ndata = sum(hist)

    ! compute multiplicities 
    ! nmax is the largest number of samples in a bin
    nbins = size(hist)
    nmax = maxval(hist)
    allocate(multiplicities(nmax+1),stat=err)
    multiplicities = 0
    multiplicities(1) = alphabet_size - nbins
    do i = 1,nbins
       ni = hist(i)
       multiplicities(ni+1) = multiplicities(ni+1) + 1
    end do

     ! working arrays on bins visited at leat once 
    ngtz = count(multiplicities > 0) 
    allocate(fgtz(ngtz),stat=err)
    allocate(multi_gtz(ngtz),stat=err)

    k = 0
    do i = 1,nmax+1
       if (multiplicities(i) > 0) then 
          k = k + 1
          fgtz(k) = i
          multi_gtz(k) = multiplicities(i)
       end if
    end do
    deallocate(multiplicities)
    
  end subroutine dirichlet_initialize

  subroutine dirichlet_finalize()

    deallocate(fgtz,multi_gtz)

  end subroutine dirichlet_finalize

  pure real(real64) function log_fpxa(alpha) 
    ! log(p(x|a)) (log of) marginal probability of data given alpha
    ! computed from histogram multiplicities 
    use constants
    
    real(real64), intent(in) :: alpha
    integer(int32) :: i
    real(real64)   :: a(ngtz)

    log_fpxa = log_gamma(ndata + one) + log_gamma(alpha * alphabet_size) & 
         - alphabet_size * log_gamma(alpha) - log_gamma(ndata + alpha * alphabet_size)
    do i = 1,ngtz 
       a(i) = multi_gtz(i) * (log_gamma(fgtz(i) - one + alpha) - log_gamma(fgtz(i) * one))
    end do
    log_fpxa = log_fpxa + sum(a)
    
  end function log_fpxa

  real(real64) function hdir(alpha)
    ! posterior mean entropy given the data and alpha
    ! computed from histogram multiplicities 
    use gamma_funcs, only: digamma
    use constants
    
    real(real64), intent(in) :: alpha
    integer(int32) :: i
    real(real64)   :: a(ngtz)

    hdir = 0.0_real64
    do i = 1,ngtz 
       a(i) = - multi_gtz(i) * (fgtz(i) - one + alpha) * digamma(fgtz(i) + alpha) 
    end do
    hdir = sum(a)
    hdir = hdir / (ndata + alpha * alphabet_size)
    hdir = hdir + digamma(ndata + alpha * alphabet_size + 1.0_real64)

  end function hdir

end module dirichlet_mod

module nsb_mod
  use iso_fortran_env
  implicit none

  real(real64) :: alpha1 = -20.0_real64
  real(real64) :: alpha2 =  10.0_real64
  real(real64) :: amax
  real(real64) :: log_fpxa_amax
  real(real64) :: fpa_amax

contains

  elemental real(real64) function fpa(a) 
    ! p(alpha) prior for alpha in NSB estimator
    use constants
    use gamma_funcs, only: trigamma
    use dirichlet_mod, only: alphabet_size
    
    real(real64), intent(in) :: a
    
    fpa = alphabet_size * trigamma(alphabet_size*a + one) - trigamma(a + one)
    
  end function fpa

  elemental real(real64) function fwa(alpha)
    ! un-normalized weight for alpha in the integrals; prop. to p(alpha|x)
    use dirichlet_mod, only: log_fpxa

    real(real64), intent(in) :: alpha

    fwa = fpa(alpha) * exp(log_fpxa(alpha) - log_fpxa_amax) / fpa_amax

  end function fwa

  elemental real(real64) function lfwa(alpha)
    ! un-normalized weight for alpha in the integrals; prop. to p(alpha|x)
    use dirichlet_mod, only: log_fpxa

    real(real64), intent(in) :: alpha

    lfwa = log(fpa(alpha)) - log(fpa_amax) + log_fpxa(alpha) - log_fpxa_amax

  end function lfwa

  subroutine compute_integration_range()
    use dirichlet_mod, only: log_fpxa
    
    integer(int32),parameter :: nx = 100
    real(real64)             :: dx,largest,small_number
    real(real64)             :: xs(nx),fxs(nx)
    real(real64)             :: a1,a2,f,x
    integer(int32)           :: i
    
    largest = huge(dx)
    small_number = 1.0e-30_real64
    alpha1 = -20.0_real64
    alpha2 = 10.0_real64

    amax = 1.0_real64
    log_fpxa_amax = log_fpxa(amax)
    fpa_amax = fpa(amax)

    ! find maximum
    dx = (alpha2 - alpha1) / (nx - 1.0_real64)
    do i = 1,nx
       xs(i) = alpha1 + (i - 1) * dx
    end do
    xs = exp(xs)
    
    fxs = lfwa(xs)
    i = maxloc(fxs,1,fxs < largest)
    amax = xs(i)
    log_fpxa_amax = log_fpxa(amax)
    fpa_amax = fpa(amax)

    ! recompute fxs
    fxs = exp(lfwa(xs))

    ! recompute integration range
    a1 = alpha2
    a2 = alpha1
    do i = 1,nx
       x = log(xs(i))
       f = fxs(i)
       if (x > alpha1 .and. x < a1 .and. f > small_number) a1 = log(xs(i-1))
       if (x > a2 .and. x < alpha2 .and. f > small_number) a2 = log(xs(i+1))
    end do
    
    alpha1 = a1
    alpha2 = a2 

    ! recompute the maximum
    dx = (alpha2 - alpha1) / (nx * 1.0_real64)
    do i = 1,nx
       xs(i) = alpha1 + (i - 0.5_real64) * dx
    end do
    xs = exp(xs)
    
    fxs = lfwa(xs)
    i = maxloc(fxs,1,fxs < largest)
    amax = xs(i)
    log_fpxa_amax = log_fpxa(amax)
    fpa_amax = fpa(amax)
    
  end subroutine compute_integration_range

  real(real64) function m_func(x)
    ! integrating after change of variable x = log(alpha) in the integral(s)
    use dirichlet_mod, only: hdir

    real(real64), intent(in) :: x
    real(real64) :: a

    a = exp(x)
    m_func = fwa(a) * hdir(a) * a
    
  end function m_func

  real(real64) function m2_func(x)
    ! integrating after change of variable x = log(alpha) in the integral(s)
    use dirichlet_mod, only: hdir

    real(real64), intent(in) :: x
    real(real64) :: a

    a = exp(x)
    m2_func = fwa(a) * hdir(a)**2 * a
    
  end function m2_func

  real(real64) function nrm_func(x)
    ! integrating after change of variable x = log(alpha) in the integral(s)

    real(real64), intent(in) :: x
    real(real64) :: a

    a = exp(x)
    nrm_func = fwa(a)  * a
    
  end function nrm_func

  subroutine hnsb(estimate,err_estimate) 

    real(real64), intent(out) :: estimate,err_estimate
    real(real64)              :: rslt,nrm

    nrm = quad(nrm_func,alpha1,alpha2)

    estimate = quad(m_func,alpha1,alpha2)
    estimate = estimate / nrm
    
    err_estimate = quad(m2_func,alpha1,alpha2)
    err_estimate = err_estimate / nrm
    err_estimate = sqrt(err_estimate - estimate**2)    

  end subroutine hnsb

  real(real64) function quad(func,a1,a2)
    ! wrapper to dqag routine 
    use quadrature, only: dqag

    real(real64),    external :: func
    real(real64),  intent(in) :: a1,a2 
    integer(int32), parameter :: limit = 500
    integer(int32), parameter :: lenw = 4 * limit
    real(real64)              :: abserr
    real(real64),   parameter :: epsabs = 0.0_real64
    real(real64),   parameter :: epsrel = 0.001_real64
    integer(int32)            :: ier
    integer(int32)            :: iwork(limit)
    integer(int32), parameter :: key = 6
    integer(int32)            :: last
    integer(int32)            :: neval
    real(real64),   parameter :: r8_pi = 3.141592653589793_real64
    real(real64)              :: work(lenw)

    call dqag ( func, a1, a2, epsabs, epsrel, key, quad, abserr, neval, ier, &
         limit, lenw, last, iwork, work )
    
  end function quad

end module nsb_mod

subroutine plugin(hist,estimate)
  ! plugin estimator - no prior, no regularization 
  use iso_fortran_env
  implicit none

  integer(int32), intent(in) :: hist(:)
  real(real64),  intent(out) :: estimate
  integer(int32) :: nbins
  integer(int32) :: i
  real(real64)   :: ni,ndata

  nbins = size(hist)
  if (nbins == 1) then 
     estimate = 0.0_real64
     return
  end if
  ndata = sum(hist)*1.0_real64
  estimate = 0.0_real64
  do i = 1,nbins
     ni = hist(i)*1.0_real64
     if (ni > 0) estimate = estimate - ni*log(ni)
  end do
  estimate = estimate / ndata + log(ndata)

end subroutine plugin

subroutine pseudo(hist,nc,alpha,estimate)
  use iso_fortran_env
  ! pseudocount estimator(s)
  ! estimate the bin frequencies using pseudocounts 
  ! and then compute the entropy of the regularized histogram 
  ! 
  ! connection to Bayesian modeling with a Dirichlet prior: 
  ! using a Dirichlet prior with parameter alpha, 
  ! the resulting posterior is again Dirichlet with mean corresponding to 
  ! the regularized empirical histogram with alpha as bin pseudocounts
  ! 
  ! the alpha parameter determines the specifical prior: 
  ! 0   : maximum likelihood (ML), or plugin, estimator
  ! 1/2 : Jeffreys' or Krychevsky-Trofimov (KT) estimator
  ! 1   : Laplace (LA) estimator
  ! 1/k : (where k is the number of classes) Schurmann-Grassberger (SG)  estimator
  implicit none

  integer(int32), intent(in)  :: hist(:)
  integer(int32), intent(in)  :: nc
  real(real64),   intent(in)  :: alpha
  real(real64),   intent(out) :: estimate
  integer(int32) :: nbins,ndata
  integer(int32) :: i
  real(real64)   :: ni

  nbins = size(hist)
  if (nbins == 1) then 
     estimate = 0.0_real64
     return
  end if
  ndata = sum(hist)
  estimate = 0.0_real64
  do i = 1,nbins
     ni = hist(i) + alpha
     estimate = estimate - ni*log(ni)
  end do
  if (nc > nbins) estimate = estimate - (nc - nbins)*alpha*log(alpha)
  estimate = estimate / (ndata + nc*alpha) + log(ndata + nc*alpha)

end subroutine pseudo

subroutine dirichlet(hist,nc,alpha,estimate)
  ! posterior mean entropy (averaged over Dirichlet distribution) given alpha 
  use iso_fortran_env
  use dirichlet_mod, only: dirichlet_initialize,dirichlet_finalize
  use dirichlet_mod, only: hdir
  implicit none

  integer(int32), intent(in)  :: hist(:)
  integer(int32), intent(in)  :: nc
  real(real64),   intent(in)  :: alpha
  real(real64),   intent(out) :: estimate

  if (size(hist) == 1) then 
     estimate = 0.0_real64
     return
  end if

  call dirichlet_initialize(nc,hist)

  estimate = hdir(alpha)

  call dirichlet_finalize()

end subroutine dirichlet

subroutine nsb(hist,nc,estimate,err_estimate)
  use iso_fortran_env
  use dirichlet_mod, only: dirichlet_initialize,dirichlet_finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer(int32), intent(in)  :: hist(:)
  integer(int32), intent(in)  :: nc
  real(real64),   intent(out) :: estimate
  real(real64),   intent(out) :: err_estimate

  if (size(hist) == 1) then 
     estimate = 0.0_real64
     err_estimate = 0.0_real64
     return
  end if

  call dirichlet_initialize(nc,hist)

  call compute_integration_range()

  call hnsb(estimate,err_estimate)

  call dirichlet_finalize()

end subroutine nsb

