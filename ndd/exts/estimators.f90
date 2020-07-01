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


module counter
  use constants
  implicit none
  integer, allocatable :: nk(:)
  real(float64), allocatable :: zk(:)
  real(float64) :: n_data
  real(float64) :: n_bins
  real(float64) :: k1

contains

  subroutine counts_reset()
    implicit none
    n_data = 0.0_float64
    n_bins = 0.0_float64
    k1 = 0.0_float64
    if (allocated(nk)) then
       deallocate(nk)
    end if
    if (allocated(zk)) then
       deallocate(zk)
    end if
  end subroutine counts_reset

  subroutine fit(n, ar)
    ! Exposed to Python
    integer, intent(in) :: n
    integer, intent(in) :: ar(n)
    call counts_fit(ar)
  end subroutine fit

  subroutine fit_zk(n, ar, br)
    ! Exposed to Python
    integer, intent(in) :: n
    integer, intent(in) :: ar(n)
    integer, intent(in) :: br(n)
    call counts_fit(ar, br)
  end subroutine fit_zk

  subroutine counts_fit(ar, br)
    implicit none
    integer, intent(in) :: ar(:)
    integer, intent(in), optional :: br(:)
    integer :: i, j, u
    integer :: x, err
    integer :: xmax
    integer, allocatable :: wrk(:)

    call counts_reset()

    if (present(br)) then  ! multiplicities
       allocate(nk(size(ar)), stat=err)
       allocate(zk(size(br)), stat=err)
       nk = ar
       zk = br
    else
       xmax = maxval(ar)
       allocate(wrk(0:xmax), stat=err)
       wrk = 0
       u = 0
       do i = 1,ubound(ar, 1)
          x = ar(i)
          if (wrk(x) == 0) then
             u = u + 1
          end if
          wrk(x) = wrk(x) + 1
       end do

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
       deallocate(wrk)
    end if

    n_data = sum(zk * nk)
    n_bins = sum(zk)
    k1 = sum(zk, nk>0)

  end subroutine counts_fit

  subroutine add_empty_bins(alphabet_size)
    ! add empty bins to nk, zk
    implicit none
    real(float64), intent(in) :: alphabet_size
    integer :: k, err
    real(float64) :: unobserved
    integer, allocatable :: wrk(:)

    if (allocated(nk)) then
       if (alphabet_size > n_bins) then
          unobserved = alphabet_size - n_bins*1.0_float64
          if (minval(nk) ==  0) then
             k = minloc(nk, 1)
             zk(k) = zk(k) + unobserved
          else
             k = size(nk)
             allocate(wrk(k), stat=err)

             wrk(:k) = nk
             deallocate(nk)
             allocate(nk(k+1), stat=err)
             nk(:k) = wrk
             nk(k+1) = 0.0_float64

             wrk(:k) = zk
             deallocate(zk)
             allocate(zk(k+1), stat=err)
             zk(:k) = wrk
             zk(k+1) = unobserved

             deallocate(wrk)
          end if
       end if
    end if

  end subroutine add_empty_bins
end module counter


module dirichlet_mod
  use constants
  use counter
  implicit none

  real(float64)              :: alphabet_size
  real(float64), allocatable :: phi(:)  ! wrk array for var

contains

  subroutine initialize(counts, nc, zk1)
    ! set n_multi, nk, multi
    integer, intent(in) :: counts(:)
    real(float64), intent(in) :: nc
    integer, intent(in), optional :: zk1(:)
    integer :: err

    alphabet_size = nc

    if (present(zk1)) then
       call counts_fit(counts, zk1)
    else
       call counts_fit(counts)
    end if

    call add_empty_bins(alphabet_size)

    allocate(phi(size(nk)), stat=err)

  end subroutine initialize

  subroutine finalize()

    if (allocated(zk)) then
       deallocate(zk)
    end if

    if (allocated(nk)) then
       deallocate(nk)
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

    wsum = sum(zk * (log_gamma(nk + alpha) - log_gamma(nk + one)))

    log_pna = log_pna + wsum

  end function log_pna

  pure real(float64) function log_pna_u(alpha)
    ! log of "unnormalized" pna. keep only alpha-dependent terms

    real(float64), intent(in) :: alpha

    log_pna_u = log_gamma(alpha * alphabet_size) &
         - alphabet_size * log_gamma(alpha) &
         - log_gamma(n_data + alpha * alphabet_size) &
         + sum(zk * (log_gamma(nk + alpha)))

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

    h_dir = - sum(zk * (nk + alpha) * digamma(nk + alpha + one))
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
    phi = digamma(nk + alpha + one) - &
         digamma(nu + two)
    c = trigamma(nu + two)

    h_var = 0.0_float64
    do i_ = 1, size(zk)
       ni = nk(i_) + alpha
       xi = phi(i_)
       jsum = sum(zk * ni * (nk + alpha) * &
            (xi * phi - c))
       h_var = h_var + zk(i_) * jsum
       h_var = h_var - zk(i_) * ni**2 * (xi**2 - c)
       xi = xi + 1 / (ni + one)
       h_var = h_var + zk(i_) * (ni + one) * ni * &
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
    use dirichlet_mod, only: alphabet_size, n_data, zk,&
         nk
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

    wsum = sum(zk * (digamma(nk + alpha)))

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

subroutine plugin(n, counts, estimate)
  ! plugin estimator - no prior, no regularization
  use constants
  use counter
  implicit none

  integer, intent(in) :: n
  integer, intent(in) :: counts(n)
  real(float64),  intent(out) :: estimate

  integer :: i
  real(float64)   :: ni

  call counts_fit(counts)

  estimate = 0.0_float64
  do i = 1,size(nk)
     ni = nk(i)
     if (ni > 0) estimate = estimate - zk(i)*ni*log(ni*1.0_float64)
  end do
  estimate = estimate / n_data + log(n_data)

  call counts_reset()

end subroutine plugin

subroutine plugin_from_multiplicities(n, nk1, zk1, estimate)
  ! plugin estimator - no prior, no regularization
  use constants
  use counter
  implicit none

  integer, intent(in) :: n
  integer, intent(in) :: nk1(n)
  integer, intent(in) :: zk1(n)
  real(float64),  intent(out) :: estimate

  integer :: i
  real(float64)   :: ni

  call counts_fit(nk1, zk1)

  estimate = 0.0_float64
  do i = 1,size(nk)
     ni = nk(i)
     if (ni > 0) estimate = estimate - zk(i)*ni*log(ni*1.0_float64)
  end do
  estimate = estimate / n_data + log(n_data)

  call counts_reset()

end subroutine plugin_from_multiplicities

subroutine pseudo(n, counts, nc, alpha, estimate)
  ! pseudo counts
  ! the alpha parameter determines the specifical prior:
  ! 0   : maximum likelihood (ML), or plugin, estimator
  ! 1/2 : Jeffreys' or Krychevsky-Trofimov (KT) estimator
  ! 1   : Laplace (LA) estimator
  ! 1/k : (where k is the number of classes) Schurmann-Grassberger (SG)  estimator
  use constants
  use counter
  implicit none

  integer, intent(in) :: n
  integer, intent(in) :: counts(n)
  real(float64), intent(in)  :: nc
  real(float64),   intent(in)  :: alpha
  real(float64), intent(out) :: estimate

  integer :: i
  real(float64)   :: ni, na

  call counts_fit(counts)

  call add_empty_bins(nc)

  estimate = 0.0_float64
  do i = 1,size(nk)
     ni = nk(i) + alpha
     estimate = estimate - zk(i)*ni*log(ni)
  end do
  na = nc * alpha
  estimate = estimate / (n_data + na) + log(n_data + na)

  call counts_reset()

end subroutine pseudo

subroutine pseudo_from_multiplicities(n, nk1, zk1, nc, alpha, estimate)
  ! pseudo counts
  use constants
  use counter
  implicit none

  integer, intent(in) :: n
  integer, intent(in) :: nk1(n)
  integer, intent(in) :: zk1(n)
  real(float64), intent(in)  :: nc
  real(float64),   intent(in)  :: alpha
  real(float64), intent(out) :: estimate

  integer :: i
  real(float64)   :: ni, na

  call counts_fit(nk1, zk1)

  call add_empty_bins(nc)

  estimate = 0.0_float64
  do i = 1,size(nk)
     ni = nk(i) + alpha
     estimate = estimate - zk(i)*ni*log(ni)
  end do
  na = nc * alpha
  estimate = estimate / (n_data + na) + log(n_data + na)

  call counts_reset()

end subroutine pseudo_from_multiplicities

subroutine ww(n, counts, nc, alpha, estimate, err_estimate)
  ! posterior mean entropy (averaged over Dirichlet distribution) given alpha
  use constants
  use dirichlet_mod, only: initialize, finalize
  use dirichlet_mod, only: h_dir, h_var
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(in)  :: alpha
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate

  call initialize(counts, nc)

  estimate = h_dir(alpha)

  err_estimate = sqrt(h_var(alpha) - estimate**2)

  call finalize()

end subroutine ww

subroutine ww_from_multiplicities(n, nk1, zk1, nc, alpha, estimate, &
  err_estimate)
  ! posterior mean entropy (averaged over Dirichlet distribution) given alpha
  use constants
  use dirichlet_mod, only: initialize, finalize
  use dirichlet_mod, only: h_dir, h_var
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)    :: nk1(n)
  integer, intent(in)    :: zk1(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(in)  :: alpha
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate

  call initialize(nk1, nc, zk1)

  estimate = h_dir(alpha)

  err_estimate = sqrt(h_var(alpha) - estimate**2)

  call finalize()

end subroutine ww_from_multiplicities

subroutine nsb(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)  :: counts(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate
  integer :: err

  call initialize(counts, nc)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call finalize()

end subroutine nsb

subroutine nsb_from_multiplicities(n, nk1, zk1, nc, estimate, err_estimate)
  use constants
  use dirichlet_mod, only: initialize, finalize
  use nsb_mod, only: hnsb
  use nsb_mod, only: compute_integration_range
  implicit none

  integer, intent(in)  :: n
  integer, intent(in)    :: nk1(n)
  integer, intent(in)    :: zk1(n)
  real(float64), intent(in)    :: nc
  real(float64),   intent(out) :: estimate
  real(float64),   intent(out) :: err_estimate
  integer :: err

  call initialize(nk1, nc, zk1)

  call compute_integration_range()

  call hnsb(estimate,err_estimate, err)

  call finalize()

end subroutine nsb_from_multiplicities

subroutine phony_1(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize, finalize
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

  call initialize(counts, nc)

  ! call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_1


subroutine phony_2(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize, finalize
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

  call initialize(counts, nc)

  ! call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_2


subroutine phony_3(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize, finalize
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

  call initialize(counts, nc)

  call compute_integration_range()

  ! call hnsb(estimate,err_estimate, err)

  call finalize()

  call cpu_time(finish)

  estimate = finish - start

end subroutine phony_3


subroutine phony_4(n,counts,nc,estimate,err_estimate)
  use constants
  use dirichlet_mod, only: initialize, finalize
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

  call initialize(counts, nc)

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
