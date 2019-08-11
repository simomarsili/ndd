! Copyright (C) 2016, Simone Marsili
! All rights reserved.
! License: BSD 3 clause

module constants
  use iso_fortran_env
  implicit none

  real(real64), parameter :: zero = 0.0_real64
  real(real64), parameter :: one = 1.0_real64
  real(real64), parameter :: two = 2.0_real64

end module constants

module dirichlet_mod
  use iso_fortran_env
  implicit none

  integer(int32)              :: n_data
  integer(int32)              :: n_multi
  real(real64)                :: n_empty_bins
  real(real64)                :: alphabet_size
  integer(int32), allocatable :: multi_z(:)
  integer(int32), allocatable :: multi(:)

contains

  subroutine initialize_dirichlet(counts, nc)
    ! set alphabet_size, n_data
    ! set n_multi, multi_z, multi
    integer(int32), intent(in) :: counts(:)
    real(real64), intent(in) :: nc

    alphabet_size = nc
    n_data = sum(counts)

    call compute_multiplicities(counts)

  end subroutine initialize_dirichlet

  subroutine compute_multiplicities(counts)
    ! set n_multi, multi_z, multi
    integer(int32), intent(in) :: counts(:)
    integer(int32)              :: nbins
    integer(int32)              :: i_,k_,ni_
    integer(int32)              :: err
    integer(int32)              :: nmax
    integer(int32), allocatable :: multi0(:)

    ! compute multiplicities
    ! nmax is the largest number of samples in a bin
    nbins = size(counts)
    nmax = maxval(counts)
    allocate(multi0(nmax),stat=err)
    ! multi0(n) is the number of states with frequency n
    multi0 = 0
    ! take into account the alphabet_size - nbins states with zero frequency
    n_empty_bins = alphabet_size - nbins
    do i_ = 1,nbins
       ni_ = counts(i_)
       if (ni_ == 0) then
          n_empty_bins = n_empty_bins + 1.0_real64
       else
          multi0(ni_) = multi0(ni_) + 1
       end if
    end do

    ! further compress data into 'sparse' multiplicities
    n_multi = count(multi0 > 0)
    allocate(multi_z(n_multi),stat=err)
    allocate(multi(n_multi),stat=err)
    k_ = 0
    do i_ = 1, nmax
       if (multi0(i_) > 0) then
          k_ = k_ + 1
          multi_z(k_) = i_
          multi(k_) = multi0(i_)
       end if
    end do
    deallocate(multi0)

  end subroutine compute_multiplicities

  subroutine dirichlet_finalize()

    deallocate(multi_z,multi)

  end subroutine dirichlet_finalize

  pure real(real64) function log_pna(alpha)
    ! log(p(n|a)) (log of) marginal probability of data given alpha
    ! computed from histogram multiplicities. Dirichlet-multinomial.
    use constants

    real(real64), intent(in) :: alpha
    integer(int32) :: i_
    real(real64)   :: wsum

    log_pna = log_gamma(n_data + one) &
         + log_gamma(alpha * alphabet_size) &
         - alphabet_size * log_gamma(alpha) &
         - log_gamma(n_data + alpha * alphabet_size)

    wsum = n_empty_bins * (log_gamma(alpha) - log_gamma(one))
    wsum = wsum + &
         sum(multi * (log_gamma(multi_z + alpha) - log_gamma(multi_z + one)))

    log_pna = log_pna + wsum

  end function log_pna

  elemental real(real64) function log_fpa(alpha)
    ! prop. to p(alpha) - the prior for alpha in NSB estimator
    use constants
    use gamma_funcs, only: trigamma

    real(real64), intent(in) :: alpha

    log_fpa = log(alphabet_size * trigamma(alphabet_size * alpha + one) - trigamma(alpha + one))

  end function log_fpa

  elemental real(real64) function log_weight(alpha)
    ! un-normalized weight for alpha in the integrals; prop. to p(alpha|x)
    real(real64), intent(in) :: alpha

    log_weight = log_fpa(alpha) + log_pna(alpha)

  end function log_weight

  real(real64) function h_bayes(alpha)
    ! posterior average of the entropy given data and a specific alpha value
    ! computed from histogram multiplicities
    use gamma_funcs, only: digamma
    use constants

    real(real64), intent(in) :: alpha
    integer(int32) :: i_

    h_bayes = - n_empty_bins * alpha * digamma(alpha + one)
    h_bayes = h_bayes - &
         sum(multi * (multi_z + alpha) * digamma(multi_z + alpha + one))
    h_bayes = h_bayes / (n_data + alpha * alphabet_size)
    h_bayes = h_bayes + digamma(n_data + alpha * alphabet_size + one)

  end function h_bayes

  elemental real(real64) function integrand(alpha, amax, order)
    ! posterior average of the entropy given the data and alpha
    ! computed from histogram multiplicities
    use gamma_funcs, only: digamma
    use constants

    real(real64), intent(in) :: alpha
    real(real64), intent(in) :: amax
    integer(int32), intent(in) :: order
    real(real64) :: hb, lw, lw_max
    real(real64) :: lpna
    integer(int32) :: mi, mzi
    integer(int32) :: i_
    real(real64) :: asum, bsum

    if (order == 0) then
       lw_max = log_weight(amax)
       integrand = exp(log_weight(alpha) - lw_max)  * alpha / amax
    else
       ! lpna = log_pna(alpha)
       lpna = log_gamma(n_data + one) &
            + log_gamma(alpha * alphabet_size) &
            - alphabet_size * log_gamma(alpha) &
            - log_gamma(n_data + alpha * alphabet_size)

       bsum = n_empty_bins * (log_gamma(alpha) - log_gamma(one))
       bsum = bsum &
            + sum(multi * (log_gamma(multi_z + alpha) &
            - log_gamma(multi_z + one)))

       asum = n_empty_bins * alpha * digamma(alpha + one)
       asum = asum + sum(multi * (multi_z + alpha) &
            * digamma(multi_z + alpha + one))

       lpna = lpna + bsum
       lw = log_fpa(alpha) + lpna

       hb = -asum
       hb = hb / (n_data + alpha * alphabet_size)
       hb = hb + digamma(n_data + alpha * alphabet_size + one)

       integrand = hb**order

       lw_max = log_weight(amax)
       integrand = integrand * exp(log_weight(alpha) - lw_max)  * alpha / amax
    end if

  end function integrand

end module dirichlet_mod

module nsb_mod
  use iso_fortran_env
  implicit none

  real(real64), parameter :: alpha1 = 1.e-8_real64
  real(real64), parameter :: alpha2 = 1.e4_real64
  real(real64) :: log_alpha1
  real(real64) :: log_alpha2
  real(real64) :: amax
  real(real64) :: lw_max
  real(real64) :: ascale

contains

  elemental real(real64) function log_weight(alpha)
    ! un-normalized weight for alpha in the integrals; prop. to p(alpha|x)
    use dirichlet_mod, only: log_pna, log_fpa

    real(real64), intent(in) :: alpha

    log_weight = log_fpa(alpha) + log_pna(alpha)

  end function log_weight

  subroutine log_weight_d(alpha, logw, dlogw)
    ! compute value and derivative of log p(a | x)
    use constants
    use gamma_funcs, only: digamma, trigamma, quadgamma
    use dirichlet_mod, only: alphabet_size, n_empty_bins, n_data, multi,&
         multi_z
    use dirichlet_mod, only: log_pna, log_fpa

    real(real64), intent(in) :: alpha
    real(real64), intent(out) :: logw, dlogw

    real(real64) :: fpa, dfpa, lpna, dlpna, wsum

    fpa = alphabet_size * trigamma(alphabet_size * alpha + one) - &
         trigamma(alpha + one)

    dfpa = alphabet_size**2 * quadgamma(alphabet_size * alpha + one) - &
         quadgamma(alpha + one)

    lpna = log_gamma(n_data + one) + log_gamma(alpha * alphabet_size) &
         - alphabet_size * log_gamma(alpha) &
         - log_gamma(n_data + alpha * alphabet_size)

    wsum = n_empty_bins * (log_gamma(alpha) - log_gamma(one))
    wsum = wsum + &
         sum(multi * (log_gamma(multi_z + alpha) - log_gamma(multi_z + one)))

    lpna = lpna + wsum

    dlpna = alphabet_size * digamma(alpha * alphabet_size) &
    - alphabet_size * digamma(alpha) &
         - alphabet_size * digamma(n_data + alpha * alphabet_size)

    wsum = n_empty_bins * digamma(alpha)
    wsum = wsum + sum(multi * (digamma(multi_z + alpha)))

    dlpna = dlpna + wsum

    logw = log(fpa) + lpna
    dlogw = dfpa / fpa + dlpna

  end subroutine log_weight_d

  subroutine compute_integration_range()
    use constants
    real(real64)             :: a1,a2,f,df,x
    integer(int32)           :: i, err

    ! initialize amax and integration range
    log_alpha1 = log(alpha1)
    log_alpha2 = log(alpha2)

    a1 = alpha1
    a2 = alpha2
    amax = -one
    do i = 1,100
       x = (a1 + a2) / two
       if (abs(a2-a1)/x < 0.001) then
          amax = x
          exit
       end if
       call log_weight_d(x, f, df)
       if (df > 0) then
          a1 = x
       else if (df < 0) then
          a2 = x
       end if
    end do

    if (amax < 0) then
       write(0, *) 'p(alpha | x) maximization didnt converge'
       stop
    end if

    lw_max = log_weight(amax)

    call weight_std(ascale, err)
    if (err > 0) ascale = 0.0 ! integration error
    if (ascale > huge(x)) then
       ascale = 0
    end if

    log_alpha1 = log(amax) - 4 * ascale
    log_alpha2 = log(amax) + 4 * ascale

    if (log_alpha1 < log(alpha1)) log_alpha1 = log(alpha1)
    if (log_alpha2 > log(alpha2)) log_alpha2 = log(alpha2)

    ! write(*, *) 'amax', amax, amx, lw_max

  end subroutine compute_integration_range

  real(real64) function m_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand

    real(real64), intent(in) :: x
    real(real64) :: alpha

    alpha = exp(x)
    m_func = integrand(alpha, amax, 1)

  end function m_func

  real(real64) function m2_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand

    real(real64), intent(in) :: x
    real(real64) :: alpha

    alpha = exp(x)
    m2_func = integrand(alpha, amax, 2)

  end function m2_func

  real(real64) function nrm_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand
    real(real64), intent(in) :: x
    real(real64) :: alpha

    alpha = exp(x)
    nrm_func = integrand(alpha, amax, 0)

  end function nrm_func

  real(real64) function std_func(x)
    ! compute the integrand of std of p(la | data)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: log_weight
    real(real64), intent(in) :: x
    real(real64) :: alpha

    alpha = exp(x)
    std_func = (x - log(amax))**2 &
         * exp(log_weight(alpha) - lw_max)  * alpha / amax

  end function std_func

  subroutine weight_std(std, err)
    real(real64), intent(out) :: std
    integer(int32), intent(out) :: err

    call quad(std_func,log_alpha1,log_alpha2, std, err)

  end subroutine weight_std

  subroutine hnsb(estimate,err_estimate, err)
    use dirichlet_mod, only: h_bayes
    real(real64), intent(out) :: estimate,err_estimate
    integer(int32), intent(out) :: err
    real(real64)              :: rslt,nrm
    integer(int32)            :: ierr

    err = 0
    if (ascale < 1.e-20) then
       estimate = h_bayes(amax)
       err_estimate = 0.0
    else
       call quad(nrm_func,log_alpha1,log_alpha2, nrm, ierr)
       err = err + ierr

       call quad(m_func,log_alpha1,log_alpha2, estimate, ierr)
       err = err + ierr

       estimate = estimate / nrm

       call quad(m2_func,log_alpha1,log_alpha2, err_estimate, ierr)
       err = err + ierr
       err_estimate = err_estimate / nrm
       err_estimate = sqrt(err_estimate - estimate**2)
    end if

  end subroutine hnsb

  subroutine quad(func,a1,a2,integral,ier)
    ! wrapper to dqag routine
    use quadrature, only: dqag

    real(real64),    external :: func
    real(real64),  intent(in) :: a1,a2
    real(real64),  intent(out) :: integral
    integer(int32), intent(out) :: ier
    integer(int32), parameter :: limit = 500
    integer(int32), parameter :: lenw = 4 * limit
    real(real64)              :: abserr
    real(real64),   parameter :: epsabs = 0.0_real64
    real(real64),   parameter :: epsrel = 0.001_real64
    integer(int32)            :: iwork(limit)
    integer(int32), parameter :: key = 6
    integer(int32)            :: last
    integer(int32)            :: neval
    real(real64),   parameter :: r8_pi = 3.141592653589793_real64
    real(real64)              :: work(lenw)

    call dqag ( func, a1, a2, epsabs, epsrel, key, integral, abserr, neval, ier, &
         limit, lenw, last, iwork, work )

  end subroutine quad

end module nsb_mod

subroutine plugin(n,counts,estimate)
  ! plugin estimator - no prior, no regularization
  use iso_fortran_env
  implicit none

  integer(int32), intent(in) :: n
  integer(int32), intent(in) :: counts(n)
  real(real64),  intent(out) :: estimate

  integer(int32) :: nbins
  integer(int32) :: i
  real(real64)   :: ni,n_data
  integer(int32)              :: mi,nmax,err
  integer(int32), allocatable :: multi0(:)
  logical :: multi = .false.

  if (multi) then
     ! using multiplicities
     nbins = size(counts)
     if (nbins == 1) then
        estimate = 0.0_real64
        return
     end if
     n_data = sum(counts)*1.0_real64
     nmax = maxval(counts)
     allocate(multi0(nmax),stat=err)
     multi0 = 0
     do i = 1,nbins
        ni = counts(i)
        if (ni == 0) cycle
        multi0(ni) = multi0(ni) + 1
     end do
     estimate = 0.0_real64
     do i = 1,nmax
        mi = multi0(i)
        if (mi > 0) estimate = estimate - mi*i*log(i*1.0_real64)
     end do
     estimate = estimate / n_data + log(n_data)
     deallocate(multi0)
  else
     ! standard implementation
     nbins = size(counts)
     if (nbins == 1) then
        estimate = 0.0_real64
        return
     end if
     n_data = sum(counts)*1.0_real64
     estimate = - sum(counts * log(counts*1.0_real64), counts>0)
     estimate = estimate / n_data + log(n_data)
  end if


end subroutine plugin

subroutine pseudo(n,counts,nc,alpha,estimate)
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

  integer(int32), intent(in)  :: n
  integer(int32), intent(in)  :: counts(n)
  integer(int32), intent(in)  :: nc
  real(real64),   intent(in)  :: alpha
  real(real64),   intent(out) :: estimate

  integer(int32) :: nbins,n_data
  integer(int32) :: i
  real(real64)   :: ni

  if (alpha < 1.0e-10_real64) then
     ! if alpha == 0.0 (no pseudocounts)
     call plugin(n, counts, estimate)
     return
  end if

  nbins = size(counts)
!  if (nbins == 1) then
!     estimate = 0.0_real64
!     return
!  end if
  n_data = sum(counts)
  estimate = 0.0_real64
  do i = 1,nbins
     ni = counts(i) + alpha
     estimate = estimate - ni*log(ni)
  end do
  ! correct for the (nc - nbins) bins with frequency alpha
  if (nc < nbins) then
     write(0,*) "nsb.pseudo: nclasses cant be < than nbins in the histogram"
     stop
  end if
  if (nc > nbins) estimate = estimate - (nc - nbins)*alpha*log(alpha)
  estimate = estimate / (n_data + nc*alpha) + log(n_data + nc*alpha)

end subroutine pseudo

subroutine dirichlet(n,counts,nc,alpha,estimate)
  ! posterior mean entropy (averaged over Dirichlet distribution) given alpha
  use iso_fortran_env
  use dirichlet_mod, only: initialize_dirichlet, compute_multiplicities, dirichlet_finalize
  use dirichlet_mod, only: h_bayes
  implicit none

  integer(int32), intent(in)  :: n
  integer(int32), intent(in)  :: counts(n)
  real(real64), intent(in)    :: nc
  real(real64),   intent(in)  :: alpha
  real(real64),   intent(out) :: estimate

!  if (size(counts) == 1) then
!     estimate = 0.0_real64
!     return
!  end if

  call initialize_dirichlet(counts, nc)
  call compute_multiplicities(counts)

  estimate = h_bayes(alpha)

  call dirichlet_finalize()

end subroutine dirichlet

subroutine nsb(n,counts,nc,estimate,err_estimate)
  use iso_fortran_env
  use dirichlet_mod, only: initialize_dirichlet, compute_multiplicities, dirichlet_finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer(int32), intent(in)  :: n
  integer(int32), intent(in)  :: counts(n)
  real(real64), intent(in)    :: nc
  real(real64),   intent(out) :: estimate
  real(real64),   intent(out) :: err_estimate
  integer(int32) :: err

  call initialize_dirichlet(counts, nc)

  call compute_multiplicities(counts)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call dirichlet_finalize()

end subroutine nsb

subroutine plugin2d(n,m,counts,estimate)
  ! plugin estimator - no prior, no regularization
  use iso_fortran_env
  implicit none

  integer(int32), intent(in) :: n
  integer(int32), intent(in) :: m
  integer(int32), intent(in) :: counts(n,m)
  real(real64),  intent(out) :: estimate(m)

  integer(int32) :: k

  do k = 1,m
     call plugin(n, counts(:,k), estimate(k))
  end do

end subroutine plugin2d

subroutine pseudo2d(n,m,counts,nc,alpha,estimate)
  use iso_fortran_env
  implicit none

  integer(int32), intent(in)  :: n
  integer(int32), intent(in)  :: m
  integer(int32), intent(in)  :: counts(n,m)
  integer(int32), intent(in)  :: nc
  real(real64),   intent(in)  :: alpha
  real(real64),   intent(out) :: estimate(m)

  integer(int32) :: nbins,n_data
  integer(int32) :: i
  real(real64)   :: ni
  integer(int32) :: k

  if (alpha < 1.0e-10_real64) then
     ! if alpha == 0.0 (no pseudocounts)
     do k = 1,m
        call plugin(n, counts(:,k), estimate(k))
     end do
  else
     do k = 1,m
        call pseudo(n,counts(:,k),nc,alpha,estimate(k))
     end do
  end if

end subroutine pseudo2d

subroutine dirichlet2d(n,m,counts,nc,alpha,estimate)
  use iso_fortran_env
  implicit none

  integer(int32), intent(in)  :: n
  integer(int32), intent(in)  :: m
  integer(int32), intent(in)  :: counts(n,m)
  real(real64), intent(in)    :: nc
  real(real64),   intent(in)  :: alpha
  real(real64),   intent(out) :: estimate(m)
  integer(int32) :: k

  do k = 1,m
     call dirichlet(n,counts(:,k),nc,alpha,estimate(k))
  end do

end subroutine dirichlet2d

subroutine nsb2d(n,m,counts,nc,estimate,err_estimate)
  use iso_fortran_env
  implicit none

  integer(int32), intent(in)  :: n
  integer(int32), intent(in)  :: m
  integer(int32), intent(in)  :: counts(n,m)
  real(real64), intent(in)    :: nc
  real(real64),   intent(out) :: estimate(m)
  real(real64),   intent(out) :: err_estimate(m)
  integer(int32) :: k

  do k = 1,m
     call nsb(n,counts(:,k),nc,estimate(k),err_estimate(k))
  end do

end subroutine nsb2d

subroutine gamma0(x, y)
  use iso_fortran_env
  use gamma_funcs, only: digamma
  implicit none
  real(real64), intent(in) :: x
  real(real64), intent(out) :: y
  y = digamma(x)
end subroutine gamma0

subroutine gamma1(x, y)
  use iso_fortran_env
  use gamma_funcs, only: trigamma
  implicit none
  real(real64), intent(in) :: x
  real(real64), intent(out) :: y
  y = trigamma(x)
end subroutine gamma1
