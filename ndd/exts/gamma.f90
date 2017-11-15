module gamma_funcs
  use iso_fortran_env
  implicit none
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
    !    Input, real(real64) X, the argument of the digamma function.
    !    0 < X.
    !
    !
    !    Output, real(real64) DIGAMMA, the value of the digamma function at X.
    !
    implicit none

    real(real64)             :: digamma
    real(real64), intent(in) :: x
    real(real64), parameter :: c = 8.5_real64
    real(real64), parameter :: euler_mascheroni = 0.57721566490153286060_real64
    real(real64) r
    real(real64) x2
    !
    !  Check the input.
    !
    if ( x <= 0.0_real64 ) then
       digamma = 0.0_real64
       return
    end if
    !
    !  Approximation for small argument.
    !
    if ( x <= 0.000001_real64 ) then
       digamma = - euler_mascheroni - 1.0_real64 / x + 1.6449340668482264365_real64 * x
       return
    end if
    !
    !  Reduce to DIGAMA(X + N).
    !
    digamma = 0.0_real64
    x2 = x

    do while ( x2 < c )
       digamma = digamma - 1.0_real64 / x2
       x2 = x2 + 1.0_real64
    end do
    !
    !  Use Stirling's (actually de Moivre's) expansion.
    !
    r = 1.0_real64 / x2

    digamma = digamma + log ( x2 ) - 0.5_real64 * r

    r = r * r

    digamma = digamma &
         - r * ( 1.0_real64 / 12.0_real64 &
         - r * ( 1.0_real64 / 120.0_real64 &
         - r * ( 1.0_real64 / 252.0_real64 &
         - r * ( 1.0_real64 / 240.0_real64 &
         - r * ( 1.0_real64 / 132.0_real64 ) ) ) ) )

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
    !    Input, real(real64) X, the argument of the trigamma function.
    !    0 < X.
    !
    !
    !    Output, real(real64) TRIGAMMA, the value of the trigamma function.
    !
    implicit none

    real(real64)              :: trigamma
    real(real64), intent(in)  :: x

    real(real64), parameter :: a = 0.0001_real64
    real(real64), parameter :: b = 5.0_real64
    real(real64), parameter :: b2 =  0.1666666667_real64
    real(real64), parameter :: b4 = -0.03333333333_real64
    real(real64), parameter :: b6 =  0.02380952381_real64
    real(real64), parameter :: b8 = -0.03333333333_real64

    real(real64) y
    real(real64) z
    !
    !  Check the input.
    !
    if ( x <= 0.0_real64 ) then
       trigamma = 0.0_real64
       return
    end if

    z = x
    !
    !  Use small value approximation if X <= A.
    !
    if ( x <= a ) then
       trigamma = 1.0_real64 / x / x
       return
    end if
    !
    !  Increase argument to ( X + I ) >= B.
    !
    trigamma = 0.0_real64

    do while ( z < b )
       trigamma = trigamma + 1.0_real64 / z / z
       z = z + 1.0_real64
    end do
    !
    !  Apply asymptotic formula if argument is B or greater.
    !
    y = 1.0_real64 / z / z

    trigamma = trigamma + 0.5_real64 * &
         y + ( 1.0_real64 &
         + y * ( b2  &
         + y * ( b4  &
         + y * ( b6  &
         + y *   b8 )))) / z

  end function trigamma
end module gamma_funcs
