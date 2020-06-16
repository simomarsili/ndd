! Copyright (C) 2016, Simone Marsili
! All rights reserved.
! License: BSD 3 clause

module constants
  implicit none

  real(8), parameter :: zero = 0.0d0
  real(8), parameter :: one = 1.0d0
  real(8), parameter :: two = 2.0d0

end module constants

module dirichlet_mod
  implicit none

  integer              :: n_data
  real(8)                :: alphabet_size
  real(8), allocatable :: hn(:)  ! array of observed frequencies
  real(8), allocatable :: hz(:)  ! multiplicities of frequency z
  real(8), allocatable :: phi(:)  ! wrk array for var

contains

  subroutine initialize_from_counts(counts, nc)
    ! set n_multi, hn, multi
    use constants
    integer, intent(in) :: counts(:)
    real(8), intent(in) :: nc
    integer              :: nbins
    integer              :: i_,k_,ni_
    integer              :: err
    integer              :: nmax
    integer, allocatable :: multi0(:)
    real(8)                :: n_empty_bins
    integer              :: n_multi

    alphabet_size = nc

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
          n_empty_bins = n_empty_bins + 1.0d0
       else
          multi0(ni_) = multi0(ni_) + 1
       end if
    end do

    ! further compress data into 'sparse' multiplicities
    n_multi = count(multi0 > 0)
    allocate(hn(n_multi+1),stat=err)
    allocate(hz(n_multi+1),stat=err)
    hn(1) = 0
    hz(1) = n_empty_bins
    k_ = 1
    do i_ = 1, nmax
       if (multi0(i_) > 0) then
          k_ = k_ + 1
          hn(k_) = i_
          hz(k_) = multi0(i_)
       end if
    end do
    deallocate(multi0)

    allocate(phi(n_multi+1), stat=err)

    n_data = sum(hz * hn)

  end subroutine initialize_from_counts

  subroutine initialize_from_multiplicities(hn1, hz1)
    ! set hn, hz, n_data, alphabet_size
    real(8), intent(in) :: hn1(:)
    real(8), intent(in) :: hz1(:)
    integer           :: err


    allocate(hn(size(hn1)), stat=err)
    allocate(hz(size(hz1)), stat=err)
    allocate(phi(size(hn1)), stat=err)
    hn = hn1
    hz = hz1

    alphabet_size = sum(hz)
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

  pure real(8) function log_pna(alpha)
    ! log(p(n|a)) (log of) marginal probability of data given alpha
    ! computed from histogram multiplicities. Dirichlet-multinomial.
    use constants

    real(8), intent(in) :: alpha
    integer :: i_
    real(8)   :: wsum

    log_pna = log_gamma(n_data + one) &
         + log_gamma(alpha * alphabet_size) &
         - alphabet_size * log_gamma(alpha) &
         - log_gamma(n_data + alpha * alphabet_size)

    wsum = sum(hz * (log_gamma(hn + alpha) - log_gamma(hn + one)))

    log_pna = log_pna + wsum

  end function log_pna

  pure real(8) function log_pna_u(alpha)
    ! log of "unnormalized" pna. keep only alpha-dependent terms
    use constants

    real(8), intent(in) :: alpha

    log_pna_u = log_gamma(alpha * alphabet_size) &
         - alphabet_size * log_gamma(alpha) &
         - log_gamma(n_data + alpha * alphabet_size) &
         + sum(hz * (log_gamma(hn + alpha)))

  end function log_pna_u


  elemental real(8) function alpha_prior(alpha)
    ! prop. to p(alpha) - the prior for alpha in NSB estimator
    use constants
    use gamma_funcs, only: trigamma

    real(8), intent(in) :: alpha

    alpha_prior = alphabet_size * trigamma(alphabet_size * alpha + one) - &
         trigamma(alpha + one)

  end function alpha_prior


  elemental real(8) function log_weight(alpha)
    ! un-normalized weight for alpha in the integrals; prop. to p(alpha|x)
    real(8), intent(in) :: alpha

    log_weight = log(alpha_prior(alpha)) + log_pna_u(alpha)

  end function log_weight

  elemental real(8) function h_dir(alpha)
    ! posterior average of the entropy given data and a specific alpha value
    ! computed from histogram multiplicities
    use gamma_funcs, only: digamma
    use constants

    real(8), intent(in) :: alpha
    integer :: i_

    h_dir = - sum(hz * (hn + alpha) * digamma(hn + alpha + one))
    h_dir = h_dir / (n_data + alpha * alphabet_size)
    h_dir = h_dir + digamma(n_data + alpha * alphabet_size + one)

  end function h_dir

  real(8) function h_var(alpha)
    ! posterior average of the entropy given data and a specific alpha value
    ! computed from histogram multiplicities
    use gamma_funcs, only: digamma, trigamma
    use constants

    real(8), intent(in) :: alpha
    integer :: i_
    real(8) :: c, nu, ni, xi, jsum

    nu = n_data + alpha * alphabet_size
    phi = digamma(hn + alpha + one) - &
         digamma(nu + two)
    c = trigamma(nu + two)

    h_var = 0.0
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


  real(8) function integrand(alpha, amax, order)
    ! posterior average of the entropy given the data and alpha
    ! computed from histogram multiplicities
    use gamma_funcs, only: digamma
    use constants

    real(8), intent(in) :: alpha
    real(8), intent(in) :: amax
    integer, intent(in) :: order
    real(8) :: hb, lw, lw_max
    real(8) :: lpna
    integer :: mi, mzi
    integer :: i_
    real(8) :: asum, bsum

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
  implicit none

  real(8), parameter :: alpha1 = 1.d-8
  real(8), parameter :: alpha2 = 1.d4
  real(8) :: log_alpha1
  real(8) :: log_alpha2
  real(8) :: amax
  real(8) :: ascale

contains

  elemental real(8) function log_weight(alpha)
    ! un-normalized weight for alpha in the integrals; prop. to p(alpha|x)
    use dirichlet_mod, only: log_pna_u, alpha_prior

    real(8), intent(in) :: alpha

    log_weight = log(alpha_prior(alpha)) + log_pna_u(alpha)

  end function log_weight

  subroutine log_weight_d(alpha, logw, dlogw)
    ! compute value and derivative of log p(a | x)
    use constants
    use gamma_funcs, only: digamma, trigamma, quadgamma
    use dirichlet_mod, only: alphabet_size, n_data, hz,&
         hn
    use dirichlet_mod, only: log_pna_u, alpha_prior

    real(8), intent(in) :: alpha
    real(8), intent(out) :: logw, dlogw

    real(8) :: prior, dprior, lpna, dlpna, wsum

    ! log weight
    prior = alpha_prior(alpha)
    logw = log(prior) + log_pna_u(alpha)

    ! log weight derivative
    dprior = alphabet_size**2 * quadgamma(alphabet_size * alpha + one) - &
         quadgamma(alpha + one)

    dlpna = alphabet_size * digamma(alpha * alphabet_size) &
    - alphabet_size * digamma(alpha) &
         - alphabet_size * digamma(n_data + alpha * alphabet_size)

    wsum = sum(hz * (digamma(hn + alpha)))

    dlpna = dlpna + wsum

    dlogw = dprior / prior + dlpna

  end subroutine log_weight_d

  subroutine compute_integration_range()
    use constants
    real(8)             :: a1,a2,f,df,x
    integer           :: i, err

    ! initialize amax and integration range
    log_alpha1 = log(alpha1)
    log_alpha2 = log(alpha2)

    ! find the location of the maximum of log w(alpha) = log p(alpha | x)
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

    call weight_std(ascale, err)
    if (err > 0) ascale = 0.0 ! integration error
    if (ascale > huge(x)) then
       ascale = 0
    end if

    log_alpha1 = log(amax) - 4 * ascale
    log_alpha2 = log(amax) + 4 * ascale

    ! check integration boundaries
    if (log_alpha1 < log(alpha1)) log_alpha1 = log(alpha1)
    if (log_alpha2 > log(alpha2)) log_alpha2 = log(alpha2)

  end subroutine compute_integration_range

  real(8) function m_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand

    real(8), intent(in) :: x
    real(8) :: alpha

    alpha = exp(x)
    m_func = integrand(alpha, amax, 1)

  end function m_func

  real(8) function m2_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand

    real(8), intent(in) :: x
    real(8) :: alpha

    alpha = exp(x)
    m2_func = integrand(alpha, amax, 2)

  end function m2_func

  real(8) function nrm_func(x)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: integrand
    real(8), intent(in) :: x
    real(8) :: alpha

    alpha = exp(x)
    nrm_func = integrand(alpha, amax, 0)

  end function nrm_func

  real(8) function var_func(x)
    ! compute the integrand of std of p(la | data)
    ! integrate over x = log(alpha)
    use dirichlet_mod, only: log_weight
    real(8), intent(in) :: x
    real(8) :: alpha

    alpha = exp(x)
    var_func = (x - log(amax))**2 &
         * exp(log_weight(alpha) - log_weight(amax))  * alpha / amax

  end function var_func

  subroutine weight_std(std, err)
    real(8), intent(out) :: std
    integer, intent(out) :: err
    real(8) :: var, nrm

    call quad(var_func,log_alpha1,log_alpha2, var, err)
    call quad(nrm_func,log_alpha1,log_alpha2, nrm, err)
    std = sqrt(var/nrm)

    if (isnan(std)) then
       err = 1
    end if

  end subroutine weight_std

  subroutine hnsb(estimate,err_estimate, err)
    use dirichlet_mod, only: h_dir, h_var
    real(8), intent(out) :: estimate,err_estimate
    integer, intent(out) :: err
    real(8)              :: rslt,nrm
    integer            :: ierr

    err = 0
    if (ascale < 1.e-20) then
       estimate = h_dir(amax)
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
       if (isnan(err_estimate)) then
          err_estimate = 0.0d0
       end if
    end if

  end subroutine hnsb

  subroutine quad(func,a1,a2,integral,ier)
    ! wrapper to dqag routine
    use quadrature, only: dqag

    real(8),    external :: func
    real(8),  intent(in) :: a1,a2
    real(8),  intent(out) :: integral
    integer, intent(out) :: ier
    integer, parameter :: limit = 500
    integer, parameter :: lenw = 4 * limit
    real(8)              :: abserr
    real(8),   parameter :: epsabs = 0.0d0
    real(8),   parameter :: epsrel = 0.001d0
    integer            :: iwork(limit)
    integer, parameter :: key = 6
    integer            :: last
    integer            :: neval
    real(8),   parameter :: r8_pi = 3.141592653589793d0
    real(8)              :: work(lenw)

    call dqag ( func, a1, a2, epsabs, epsrel, key, integral, abserr, neval, ier, &
         limit, lenw, last, iwork, work )

  end subroutine quad

end module nsb_mod

subroutine plugin(n,counts,estimate)
  ! plugin estimator - no prior, no regularization
  implicit none

  integer, intent(in) :: n
  integer, intent(in) :: counts(n)
  real(8),  intent(out) :: estimate

  integer :: nbins
  integer :: i
  real(8)   :: ni,n_data
  integer              :: mi,nmax,err
  integer, allocatable :: multi0(:)
  logical :: multi = .false.

  if (multi) then
     ! using multiplicities
     nbins = size(counts)
     if (nbins == 1) then
        estimate = 0.0d0
        return
     end if
     n_data = sum(counts)*1.0d0
     nmax = maxval(counts)
     allocate(multi0(nmax),stat=err)
     multi0 = 0
     do i = 1,nbins
        ni = counts(i)
        if (ni == 0) cycle
        multi0(ni) = multi0(ni) + 1
     end do
     estimate = 0.0d0
     do i = 1,nmax
        mi = multi0(i)
        if (mi > 0) estimate = estimate - mi*i*log(i*1.0d0)
     end do
     estimate = estimate / n_data + log(n_data)
     deallocate(multi0)
  else
     ! standard implementation
     nbins = size(counts)
     if (nbins == 1) then
        estimate = 0.0d0
        return
     end if
     n_data = sum(counts)*1.0d0
     estimate = - sum(counts * log(counts*1.0d0), counts>0)
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
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  integer, intent(in)  :: nc
  real(8),   intent(in)  :: alpha
  real(8),   intent(out) :: estimate

  integer :: nbins,n_data
  integer :: i
  real(8)   :: ni

  if (alpha < 1.0d-10) then
     ! if alpha == 0.0 (no pseudocounts)
     call plugin(n, counts, estimate)
     return
  end if

  nbins = size(counts)
!  if (nbins == 1) then
!     estimate = 0.0d0
!     return
!  end if
  n_data = sum(counts)
  estimate = 0.0d0
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
  use dirichlet_mod, only: initialize_from_counts, finalize
  use dirichlet_mod, only: h_dir
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(8), intent(in)    :: nc
  real(8),   intent(in)  :: alpha
  real(8),   intent(out) :: estimate

!  if (size(counts) == 1) then
!     estimate = 0.0d0
!     return
!  end if

  call initialize_from_counts(counts, nc)

  estimate = h_dir(alpha)

  call finalize()

end subroutine dirichlet

subroutine nsb(n,counts,nc,estimate,err_estimate)
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(8), intent(in)    :: nc
  real(8),   intent(out) :: estimate
  real(8),   intent(out) :: err_estimate
  integer :: err

  call initialize_from_counts(counts, nc)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call finalize()

end subroutine nsb

subroutine nsb_from_multiplicities(n, hn1, hz1, estimate, err_estimate)
  use dirichlet_mod, only: initialize_from_multiplicities, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  real(8), intent(in)    :: hn1(n)
  real(8), intent(in)    :: hz1(n)
  real(8),   intent(out) :: estimate
  real(8),   intent(out) :: err_estimate
  integer :: err

  call initialize_from_multiplicities(hn1, hz1)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call finalize()

end subroutine nsb_from_multiplicities

subroutine phony_1(n,counts,nc,estimate,err_estimate)
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(8), intent(in)    :: nc
  real(8),   intent(out) :: estimate
  real(8),   intent(out) :: err_estimate
  integer :: err
  real :: start, finish

  call cpu_time(start)

  call initialize_from_counts(counts, nc)

  ! call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_1


subroutine phony_2(n,counts,nc,estimate,err_estimate)
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(8), intent(in)    :: nc
  real(8),   intent(out) :: estimate
  real(8),   intent(out) :: err_estimate
  integer :: err
  real :: start, finish

  call cpu_time(start)

  call initialize_from_counts(counts, nc)

  ! call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_2


subroutine phony_3(n,counts,nc,estimate,err_estimate)
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(8), intent(in)    :: nc
  real(8),   intent(out) :: estimate
  real(8),   intent(out) :: err_estimate
  integer :: err
  real :: start, finish

  call cpu_time(start)

  call initialize_from_counts(counts, nc)

  call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_3


subroutine phony_4(n,counts,nc,estimate,err_estimate)
  use dirichlet_mod, only: initialize_from_counts, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(8), intent(in)    :: nc
  real(8),   intent(out) :: estimate
  real(8),   intent(out) :: err_estimate
  integer :: err
  real :: start, finish

  call cpu_time(start)

  call initialize_from_counts(counts, nc)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_4


subroutine plugin2d(n,m,counts,estimate)
  ! plugin estimator - no prior, no regularization
  implicit none

  integer, intent(in) :: n
  integer, intent(in) :: m
  integer, intent(in) :: counts(n,m)
  real(8),  intent(out) :: estimate(m)

  integer :: k

  do k = 1,m
     call plugin(n, counts(:,k), estimate(k))
  end do

end subroutine plugin2d

subroutine pseudo2d(n,m,counts,nc,alpha,estimate)
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: m
  integer, intent(in)  :: counts(n,m)
  integer, intent(in)  :: nc
  real(8),   intent(in)  :: alpha
  real(8),   intent(out) :: estimate(m)

  integer :: nbins,n_data
  integer :: i
  real(8)   :: ni
  integer :: k

  if (alpha < 1.0d-10) then
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
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: m
  integer, intent(in)  :: counts(n,m)
  real(8), intent(in)    :: nc
  real(8),   intent(in)  :: alpha
  real(8),   intent(out) :: estimate(m)
  integer :: k

  do k = 1,m
     call dirichlet(n,counts(:,k),nc,alpha,estimate(k))
  end do

end subroutine dirichlet2d

subroutine nsb2d(n,m,counts,nc,estimate,err_estimate)
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: m
  integer, intent(in)  :: counts(n,m)
  real(8), intent(in)    :: nc
  real(8),   intent(out) :: estimate(m)
  real(8),   intent(out) :: err_estimate(m)
  integer :: k

  do k = 1,m
     call nsb(n,counts(:,k),nc,estimate(k),err_estimate(k))
  end do

end subroutine nsb2d

subroutine gamma0(x, y)
  use gamma_funcs, only: digamma
  implicit none
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = digamma(x)
end subroutine gamma0

subroutine gamma1(x, y)
  use gamma_funcs, only: trigamma
  implicit none
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = trigamma(x)
end subroutine gamma1
