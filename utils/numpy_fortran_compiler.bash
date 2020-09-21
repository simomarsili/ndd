#!/bin/bash

# get numpy lapack lib
lapack_lib=`python -c "import os; import numpy; print(os.path.abspath(numpy.linalg.lapack_lite.__file__))"`

libs=`ldd $lapack_lib`

if [[ $libs == *"libgfortran"* ]]; then
    echo "gfortran"
elif [[ $libs == *"libg2c"* ]]; then
    echo "g77"
else
    echo "not in (gfortran, g77)"
fi
