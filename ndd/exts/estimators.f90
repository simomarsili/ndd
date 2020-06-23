! Copyright (C) 2016, Simone Marsili
! All rights reserved.
! License: BSD 3 clause

module constants
  implicit none
  integer, parameter :: int32 = kind(1)
  integer, parameter :: float32 = kind(1.0)
  integer, parameter :: float64 = kind(1.d0)
  real(float64), parameter :: zero = 0.0_float64
  real(float64), parameter :: one = 1.0_float64
  real(float64), parameter :: two = 2.0_float64

end module constants

module dirichlet_mod
  use constants
  implicit none

  integer              :: n_data
  real(float64)                :: alphabet_size
  real(float64), allocatable :: hn(:)  ! array of observed frequencies
  real(float64), allocatable :: hz(:)  ! multiplicities of frequency z
  real(float64), allocatable :: phi(:)  ! wrk array for var

contains

  subroutine initialize_from_counts(counts, nc)
    ! set n_multi, hn, multi
    integer, intent(in) :: counts(:)
    real(float64), intent(in) :: nc
    integer              :: nbins
    integer              :: i_,k_,ni_
    integer              :: err
    integer              :: nmax
    integer, allocatable :: wrk(:)
    real(float64)                :: n_empty_bins
    integer              :: n_multi

    alphabet_size = nc

    ! compute multiplicities
    ! nmax is the largest number of samples in a bin
    nbins = size(counts)
    nmax = maxval(counts)
    allocate(wrk(nmax),stat=err)
    ! wrk(n) is the number of states with frequency n
    wrk = 0
    ! take into account the alphabet_size - nbins states with zero frequency
    n_empty_bins = alphabet_size - nbins
    do i_ = 1,nbins
       ni_ = counts(i_)
       if (ni_ == 0) then
          n_empty_bins = n_empty_bins + 1.0_float64
       else
          wrk(ni_) = wrk(ni_) + 1
       end if
    end do

    ! further compress data into 'sparse' multiplicities
    n_multi = count(wrk > 0)
    allocate(hn(n_multi+1),stat=err)
    allocate(hz(n_multi+1),stat=err)
    hn(1) = 0
    hz(1) = n_empty_bins
    k_ = 1
    do i_ = 1, nmax
       if (wrk(i_) > 0) then
          k_ = k_ + 1
          hn(k_) = i_
          hz(k_) = wrk(i_)
       end if
    end do
    deallocate(wrk)

    allocate(phi(n_multi+1), stat=err)

    n_data = sum(hz * hn)

  end subroutine initialize_from_counts

  subroutine initialize_from_multiplicities(hn1, hz1, nc)
    ! set hn, hz, n_data, alphabet_size
    real(float64), intent(in) :: hn1(:)
    real(float64), intent(in) :: hz1(:)
    real(float64), intent(in) :: nc
    integer :: idx=-1, err, j, nm

    nm = size(hn1)
    ! check if zeros are included in mults arrays
    if (any(int(hn1) == 0)) then  ! we work with floats
       allocate(hn(nm), stat=err)
       allocate(hz(nm), stat=err)
       allocate(phi(nm), stat=err)
       hn = hn1
       hz = hz1
       idx = -1
       do j = 1, nm
          if (int(hn1(j)) == 0) then
             idx = j
             exit
          end if
       end do
       hz(idx) = hz1(idx) + nc - sum(hz1)
    else
       allocate(hn(nm + 1), stat=err)
       allocate(hz(nm + 1), stat=err)
       allocate(phi(nm + 1), stat=err)
       hn(1) = 0._float64
       hz(1) = nc - sum(hz1)
       hn(2:) = hn1
       hz(2:) = hz1
    end if

    alphabet_size = nc
    n_data = sum(hz * hn)

  end subroutine initialize_from_multiplicities

  subroutine finalize()

    if (allocated(hz)) then
       deallocate(hz)
    end if

    if (allocated(hn)) then
       deallocate(hn)
    end if

    if (allocated(phi)) then
       deallocate(phi)
    end if

  end subroutine finalize

  pure real(float64) function log_pna(alpha)
    ! log(p(n|a)) (log of) marginal probability of data given alpha
    ! computed from histogram multiplicities. Dirichlet-multinomial.

    real(float64), intent(in) :: alpha
    integer :: i_
    real(float64)   :: wsum

    log_pna = log_gamma(n_data + one) &
         + log_gamma(alpha * alphabet_size) &
         - alphabet_size * log_gamma(alpha) &
         - log_gamma(n_data + alpha * alphabet_size)

    wsum = sum(hz * (log_gamma(hn + alpha) - log_gamma(hn + one)))

    log_pna = log_pna + wsum

  end function log_pna

  pure real(float64) function log_pna_u(alpha)
    ! log of "unnormalized" pna. keep only alpha-dependent terms

    real(float64), intent(in) :: alpha

    log_pna_u = log_gamma(alpha * alphabet_size) &
         - alphabet_size * log_gamma(alpha) &
         - log_gamma(n_data + alpha * alphabet_size) &
         + sum(hz * (log_gamma(hn + alpha)))

  end function log_pna_u


  elemental real(float64) function alpha_prior(alpha)
    ! prop. to p(alpha) - the prior for alpha in NSB estimator
    use gamma_funcs, only: trigamma

    real(float64), intent(in) :: alpha

    alpha_prior = alphabet_size * trigamma(alphabet_size * alpha + one) - &
         trigamma(alpha + one)

  end function alpha_prior


  elemental real(float64) function log_weight(alpha)
    ! un-normalized weight for alpha in the integrals; prop. to p(alpha|x)
    real(float64), intent(in) :: alpha

    log_weight = log(alpha_prior(alpha)) + log_pna_u(alpha)

  end function log_weight

  elemental real(float64) function h_dir(alpha)
    ! posterior average of the entropy given data and a specific alpha value
    ! computed from histogram multiplicities
    use gamma_funcs, only: digamma

    real(float64), intent(in) :: alpha
    integer :: i_

    h_dir = - sum(hz * (hn + alpha) * digamma(hn + alpha + one))
    h_dir = h_dir / (n_data + alpha * alphabet_size)
    h_dir = h_dir + digamma(n_data + alpha * alphabet_size + one)

  end function h_dir

  real(float64) function h_var(alpha)
    ! posterior average of the entropy given data and a specific alpha value
    ! computed from histogram multiplicities
    use gamma_funcs, only: digamma, trigamma

    real(float64), intent(in) :: alpha
    integer :: i_
    real(float64) :: c, nu, ni, xi, jsum

    nu = n_data + alpha * alphabet_size
    phi = digamma(hn + alpha + one) - &
         digamma(nu + two)
    c = trigamma(nu + two)

    h_var = 0.0_float64
    do i_ = 1, size(hz)
       ni = hn(i_) + alpha
       xi = phi(i_)
       jsum = sum(hz * ni * (hn + alpha) * &
            (xi * phi - c))
       h_var = h_var + hz(i_) * jsum
       h_var = h_var - hz(i_) * ni**2 * (xi**2 - c)
       xi = xi + 1 / (ni + one)
       h_var = h_var + hz(i_) * (ni + one) * ni * &
            (xi**2 + trigamma(ni + two) - c)
    end do

    h_var = h_var / (nu * (nu + one))

  end function h_var


  real(float64) function integrand(alpha, amax, order)
    ! posterior average of the entropy given the data and alpha
    ! computed from histogram multiplicities
    use gamma_funcs, only: digamma

    real(float64), intent(in) :: alpha
    real(float64), intent(in) :: amax
    integer, intent(in) :: order
    real(float64) :: hb, lw, lw_max
    real(float64) :: lpna
    integer :: mi, mzi
    integer :: i_
    real(float64) :: asum, bsum

    if (order == 0) then
       lw_max = log_weight(amax)
       integrand = exp(log_weight(alpha) - lw_max)  * alpha / amax
    else
       if (order == 1) then
          integrand = h_dir(alpha)
       else if (order == 2) then
          integrand = h_var(alpha)
       end if

       lw_max = log_weight(amax)

       integrand = integrand * exp(log_weight(alpha) - lw_max)  * &
            exp(log(alpha) - log(amax))
    end if

  end function integrand

end module dirichlet_mod

module nsb_mod
  use constants
  implicit none

  real(float64) :: alpha1
  real(float64) :: alpha2
  real(float64) :: log_alpha1
  real(float64) :: log_alpha2
  real(float64) :: amax
  real(float64) :: ascale

contains

  elemental real(float64) function log_weight(alpha)
    ! un-normalized weight for alpha in the integrals; prop. to p(alpha|x)
    use dirichlet_mod, only: log_pna_u, alpha_prior

    real(float64), intent(in) :: alpha

    log_weight = log(alpha_prior(alpha)) + log_pna_u(alpha)

  end function log_weight

  subroutine log_weight_d(alpha, logw, dlogw)
    ! compute value and derivative of log p(a | x)
    use gamma_funcs, only: digamma, trigamma, quadgamma
    use dirichlet_mod, only: alphabet_size, n_data, hz,&
         hn
    use dirichlet_mod, only: log_pna_u, alpha_prior

    real(float64), intent(in) :: alpha
    real(float64), intent(out) :: logw, dlogw

    real(float64) :: prior, float64rior, lpna, dlpna, wsum

    ! log weight
    prior = alpha_prior(alpha)
    logw = log(prior) + log_pna_u(alpha)

    ! log weight derivative
    float64rior = alphabet_size**2 * quadgamma(alphabet_size * alpha + one) - &
         quadgamma(alpha + one)

    dlpna = alphabet_size * digamma(alpha * alphabet_size) &
    - alphabet_size * digamma(alpha) &
         - alphabet_size * digamma(n_data + alpha * alphabet_size)

    wsum = sum(hz * (digamma(hn + alpha)))

    dlpna = dlpna + wsum

    dlogw = float64rior / prior + dlpna

  end subroutine log_weight_d

  subroutine compute_integration_range()
    use dirichlet_mod, only: alphabet_size
    real(float64)             :: a1,a2,f,df,x
    integer           :: i, err

    amax = 1/alphabet_size
    alpha1 = amax * 1.d-8
    alpha2 = amax * 1.d8

    ! initialize amax and integration range
    log_alpha1 = log(alpha1)
    log_alpha2 = log(alpha2)

    ! find the location of the maximum of log w(alpha) = log p(alpha | x)
    a1 = alpha1
    a2 = alpha2
    amax = -one
    do i = 1,100
       x = (a1 + a2) / two
       if (abs(a2-a1)/x < 0.001_float64) then
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

    call weight_std(ascale, err)
    if (err > 0) ascale = 0.0_float64 ! integration error
    if (ascale > huge(x)) then
       ascale = 0
    end if

    log_alpha1 = log(amax) - 4 * ascale
    log_alpha2 = log(amax) + 4 * ascale

    ! check integration boundaries
    if (log_alpha1 < log(alpha1)) log_alpha1 = log(alpha1)
    if (log_alpha2 > log(alpha2)) log_alpha2 = log(alpha2)

  end subroutine compute_integration_range

  real(float64) function m_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand

    real(float64), intent(in) :: x
    real(float64) :: alpha

    alpha = exp(x)
    m_func = integrand(alpha, amax, 1)

  end function m_func

  real(float64) function m2_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand

    real(float64), intent(in) :: x
    real(float64) :: alpha

    alpha = exp(x)
    m2_func = integrand(alpha, amax, 2)

  end function m2_func

  real(float64) function nrm_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand
    real(float64), intent(in) :: x
    real(float64) :: alpha

    alpha = exp(x)
    nrm_func = integrand(alpha, amax, 0)

  end function nrm_func

  real(float64) function var_func(x)
    ! compute the integrand of std of p(la | data)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: log_weight
    real(float64), intent(in) :: x
    real(float64) :: alpha

    alpha = exp(x)
    var_func = (x - log(amax))**2 &
         * exp(log_weight(alpha) - log_weight(amax))  * alpha / amax

  end function var_func

  subroutine weight_std(std, err)
    real(float64), intent(out) :: std
    integer, intent(out) :: err
    real(float64) :: var, nrm

    call quad(var_func,log_alpha1,log_alpha2, var, err)
    call quad(nrm_func,log_alpha1,log_alpha2, nrm, err)
    std = sqrt(var/nrm)

    if (isnan(std)) then
       err = 1
    end if

  end subroutine weight_std

  subroutine hnsb(estimate,err_estimate, err)
    use dirichlet_mod, only: h_dir, h_var
    real(float64), intent(out) :: estimate,err_estimate
    integer, intent(out) :: err
    real(float64)              :: rslt,nrm
    integer            :: ierr

    err = 0
    if (ascale < 1.e-20_float64) then
       estimate = h_dir(amax)
       err_estimate = 0.0_float64
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
       if (isnan(err_estimate)) then
          err_estimate = 0.0_float64
       end if
    end if

  end subroutine hnsb

  subroutine quad(func,a1,a2,integral,ier)
    ! wrapper to dqag routine
    use quadrature, only: dqag

    real(float64),    external :: func
    real(float64),  intent(in) :: a1,a2
    real(float64),  intent(out) :: integral
    integer, intent(out) :: ier
    integer, parameter :: limit = 500
    integer, parameter :: lenw = 4 * limit
    real(float64)              :: abserr
    real(float64),   parameter :: epsabs = 0.0_float64
    real(float64),   parameter :: epsrel = 0.001_float64
    integer            :: iwork(limit)
    integer, parameter :: key = 6
    integer            :: last
    integer            :: neval
    real(float64),   parameter :: r8_pi = 3.141592653589793_float64
    real(float64)              :: work(lenw)

    call dqag ( func, a1, a2, epsabs, epsrel, key, integral, abserr, neval, ier, &
         limit, lenw, last, iwork, work )

  end subroutine quad

end module nsb_mod

subroutine plugin(n,counts,estimate)
  ! plugin estimator - no prior, no regularization
  use constants
  implicit none

  integer, intent(in) :: n
  integer, intent(in) :: counts(n)
  real(float64),  intent(out) :: estimate

  integer :: nbins
  integer :: i
  real(float64)   :: ni,n_data
  integer              :: mi,nmax,err
  integer, allocatable :: wrk(:)
  logical :: multi = .false.

  if (multi) then
     ! using multiplicities
     nbins = size(counts)
     if (nbins == 1) then
        estimate = 0.0_float64
        return
     end if
     n_data = sum(counts)*1.0_float64
     nmax = maxval(counts)
     allocate(wrk(nmax),stat=err)
     wrk = 0
     do i = 1,nbins
        ni = counts(i)
        if (ni == 0) cycle
        wrk(ni) = wrk(ni) + 1
     end do
     estimate = 0.0_float64
     do i = 1,nmax
        mi = wrk(i)
        if (mi > 0) estimate = estimate - mi*i*log(i*1.0_float64)
     end do
     estimate = estimate / n_data + log(n_data)
     deallocate(wrk)
  else
     ! standard implementation
     nbins = size(counts)
     if (nbins == 1) then
        estimate = 0.0_float64
        return
     end if
     n_data = sum(counts)*1.0_float64
     estimate = - sum(counts * log(counts*1.0_float64), counts>0)
     estimate = estimate / n_data + log(n_data)
  end if


end subroutine plugin

subroutine pseudo(n,counts,nc,alpha,estimate)
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
  use constants
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  integer, intent(in)  :: nc
  real(float64),   intent(in)  :: alpha
  real(float64),   intent(out) :: estimate

  integer :: nbins,n_data
  integer :: i
  real(float64)   :: ni

  if (alpha < 1.0e-10_float64) then
     ! if alpha == 0.0 (no pseudocounts)
     call plugin(n, counts, estimate)
     return
  end if

  nbins = size(counts)
!  if (nbins == 1) then
!     estimate = 0.0_float64
!     return
!  end if
  n_data = sum(counts)
  estimate = 0.0_float64
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

subroutine ww(n, counts, nc, alpha, estimate, err_estimate)
  ! posterior mean entropy (averaged over Dirichlet distribution) given alpha
  use constants
  use dirichlet_mod, only: initialize_from_counts, finalize
  use dirichlet_mod, only: h_dir, h_var
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(in)  :: alpha
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate

  call initialize_from_counts(counts, nc)

  estimate = h_dir(alpha)

  err_estimate = sqrt(h_var(alpha) - estimate**2)

  call finalize()

end subroutine ww

subroutine ww_from_multiplicities(n, hn1, hz1, nc, alpha, estimate, &
  err_estimate)
  ! posterior mean entropy (averaged over Dirichlet distribution) given alpha
  use constants
  use dirichlet_mod, only: initialize_from_multiplicities, finalize
  use dirichlet_mod, only: h_dir, h_var
  implicit none

  integer, intent(in)  :: n
  real(float64), intent(in)    :: hn1(n)
  real(float64), intent(in)    :: hz1(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(in)  :: alpha
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate

  call initialize_from_multiplicities(hn1, hz1, nc)

  estimate = h_dir(alpha)

  err_estimate = sqrt(h_var(alpha) - estimate**2)

  call finalize()

end subroutine ww_from_multiplicities

subroutine nsb(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate
  integer :: err

  call initialize_from_counts(counts, nc)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call finalize()

end subroutine nsb

subroutine nsb_from_multiplicities(n, hn1, hz1, nc, estimate, err_estimate)
  use constants
  use dirichlet_mod, only: initialize_from_multiplicities, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  real(float64), intent(in)    :: hn1(n)
  real(float64), intent(in)    :: hz1(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate
  integer :: err

  call initialize_from_multiplicities(hn1, hz1, nc)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call finalize()

end subroutine nsb_from_multiplicities

subroutine phony_1(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate
  integer :: err
  real(float32) :: start, finish

  call cpu_time(start)

  call initialize_from_counts(counts, nc)

  ! call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_1


subroutine phony_2(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate
  integer :: err
  real(float32) :: start, finish

  call cpu_time(start)

  call initialize_from_counts(counts, nc)

  ! call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_2


subroutine phony_3(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate
  integer :: err
  real(float32) :: start, finish

  call cpu_time(start)

  call initialize_from_counts(counts, nc)

  call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_3


subroutine phony_4(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate
  integer :: err
  real(float32) :: start, finish

  call cpu_time(start)

  call initialize_from_counts(counts, nc)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_4


subroutine nsb2d(n,m,counts,nc,estimate,err_estimate)
  use constants
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: m
  integer, intent(in)  :: counts(n,m)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate(m)
  real(float64),   intent(out) :: err_estimate(m)
  integer :: k

  do k = 1,m
     call nsb(n,counts(:,k),nc,estimate(k),err_estimate(k))
  end do

end subroutine nsb2d

subroutine gamma0(x, y)
  use constants
  use gamma_funcs, only: digamma
  implicit none
  real(float64), intent(in) :: x
  real(float64), intent(out) :: y
  y = digamma(x)
end subroutine gamma0

subroutine gamma1(x, y)
  use constants
  use gamma_funcs, only: trigamma
  implicit none
  real(float64), intent(in) :: x
  real(float64), intent(out) :: y
  y = trigamma(x)
end subroutine gamma1

module counts
  use constants
  implicit none
  integer, allocatable :: nk(:)
  integer, allocatable :: zk(:)
  integer :: ndata
  integer :: nbins
  integer :: k1
contains
  subroutine fit(n, counts)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: counts(n)
    integer :: i, j, u
    integer :: x, err
    integer :: xmax
    integer, allocatable :: wrk(:)

    xmax = maxval(counts)
    allocate(wrk(0:xmax), stat=err)
    wrk = 0
    u = 0
    do i = 1,n
       x = counts(i)
       if (wrk(x) == 0) then
          u = u + 1
       end if
       wrk(x) = wrk(x) + 1
    end do

    if (allocated(nk)) then
       deallocate(nk)
    end if
    if (allocated(zk)) then
       deallocate(zk)
    end if
    allocate(nk(u), zk(u), stat=err)

    u = 0
    do i = 0, xmax
       x = wrk(i)
       if (x > 0) then
          u = u + 1
          nk(u) = i
          zk(u) = x
       end if
    end do

    ndata = sum(zk * nk)
    nbins = sum(zk)
    k1 = sum(zk, nk>0)

    deallocate(wrk)
  end subroutine fit
end module counts

