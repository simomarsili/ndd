module gamma_funcs
  implicit none
  integer, parameter :: int32 = kind(1)
  integer, parameter :: float32 = kind(1.0)
  integer, parameter :: float64 = kind(1.d0)
contains
  elemental function digamma (x)

    !*****************************************************************************80
    !
    !! DIGAMMA calculates DIGAMMA ( X ) = d ( LOG ( GAMMA ( X ) ) ) / dX
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    20 March 2016
    !
    !  Author:
    !
    !    Original FORTRAN77 version by Jose Bernardo.
    !    FORTRAN90 version by John Burkardt.
    !    iso_fortran_env by Simone Marsili.
    !
    !  Reference:
    !
    !    Jose Bernardo,
    !    Algorithm AS 103:
    !    Psi ( Digamma ) Function,
    !    Applied Statistics,
    !    Volume 25, Number 3, 1976, pages 315-317.
    !
    !  Parameters:
    !
    !    Input, real(float64) X, the argument of the digamma function.
    !    0 < X.
    !
    !
    !    Output, real(float64) DIGAMMA, the value of the digamma function at X.
    !
    implicit none

    real(float64)             :: digamma
    real(float64), intent(in) :: x
    real(float64), parameter :: c = 8.5_float64
    real(float64), parameter :: euler_mascheroni = 0.57721566490153286060_float64
    real(float64) r
    real(float64) x2
    !
    !  Check the input.
    !
    if ( x <= 0.0_float64 ) then
       digamma = 0.0_float64
       return
    end if
    !
    !  Approximation for small argument.
    !
    if ( x <= 0.000001_float64 ) then
       digamma = - euler_mascheroni - 1.0_float64 / x + 1.6449340668482264365_float64 * x
       return
    end if
    !
    !  Reduce to DIGAMA(X + N).
    !
    digamma = 0.0_float64
    x2 = x

    do while ( x2 < c )
       digamma = digamma - 1.0_float64 / x2
       x2 = x2 + 1.0_float64
    end do
    !
    !  Use Stirling's (actually de Moivre's) expansion.
    !
    r = 1.0_float64 / x2

    digamma = digamma + log ( x2 ) - 0.5_float64 * r

    r = r * r

    digamma = digamma &
         - r * ( 1.0_float64 / 12.0_float64 &
         - r * ( 1.0_float64 / 120.0_float64 &
         - r * ( 1.0_float64 / 252.0_float64 &
         - r * ( 1.0_float64 / 240.0_float64 &
         - r * ( 1.0_float64 / 132.0_float64 ) ) ) ) )

  end function digamma

  elemental function trigamma (x)

    !*****************************************************************************80
    !
    !! TRIGAMMA calculates trigamma(x) = d^2 log(gamma(x)) / dx^2
    !
    !  Modified:
    !
    !    19 January 2008
    !
    !  Author:
    !
    !    Original FORTRAN77 version by BE Schneider.
    !    FORTRAN90 version by John Burkardt.
    !    iso_fortran_env by Simone Marsili.
    !
    !  Reference:
    !
    !    BE Schneider,
    !    Algorithm AS 121:
    !    Trigamma Function,
    !    Applied Statistics,
    !    Volume 27, Number 1, pages 97-99, 1978.
    !
    !  Parameters:
    !
    !    Input, real(float64) X, the argument of the trigamma function.
    !    0 < X.
    !
    !
    !    Output, real(float64) TRIGAMMA, the value of the trigamma function.
    !
    implicit none

    real(float64)              :: trigamma
    real(float64), intent(in)  :: x

    real(float64), parameter :: a = 0.0001_float64
    real(float64), parameter :: b = 5.0_float64
    real(float64), parameter :: b2 =  0.1666666667_float64
    real(float64), parameter :: b4 = -0.03333333333_float64
    real(float64), parameter :: b6 =  0.02380952381_float64
    real(float64), parameter :: b8 = -0.03333333333_float64

    real(float64) y
    real(float64) z
    !
    !  Check the input.
    !
    if ( x <= 0.0_float64 ) then
       trigamma = 0.0_float64
       return
    end if

    z = x
    !
    !  Use small value approximation if X <= A.
    !
    if ( x <= a ) then
       trigamma = 1.0_float64 / x / x
       return
    end if
    !
    !  Increase argument to ( X + I ) >= B.
    !
    trigamma = 0.0_float64

    do while ( z < b )
       trigamma = trigamma + 1.0_float64 / z / z
       z = z + 1.0_float64
    end do
    !
    !  Apply asymptotic formula if argument is B or greater.
    !
    y = 1.0_float64 / z / z

    trigamma = trigamma + 0.5_float64 * &
         y + ( 1.0_float64 &
         + y * ( b2  &
         + y * ( b4  &
         + y * ( b6  &
         + y *   b8 )))) / z

  end function trigamma

  elemental function quadgamma (x)
    ! computes \psi_2(x) = d^3 log(gamma(x)) / dx^3 = d^2 \psi(x) / dx^2
    ! Simone Marsili
    implicit none

    real(float64)              :: quadgamma
    real(float64), intent(in)  :: x

    real(float64), parameter :: a = 0.0001_float64
    real(float64), parameter :: b = 5.0_float64
    real(float64) z
    !  Check the input.
    if ( x <= 0.0_float64 ) then
       quadgamma = 0.0_float64
       return
    end if

    !  Use small value approximation if X <= A.
    if ( x <= a ) then
       quadgamma = - gamma(3.0_float64) / x**3
       return
    end if

    !  Increase argument to ( X + I ) >= B.
    quadgamma = 0.0_float64
    z = x
    do while ( z < b )
       quadgamma = quadgamma - 2.0_float64 / z**3
       z = z + 1.0_float64
    end do

    !  Apply asymptotic formula if argument is B or greater.
    z = 1/z
    quadgamma = quadgamma - z**2 - z**3 - 0.5*z**4

  end function quadgamma

end module gamma_funcs
