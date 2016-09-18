from numpy.distutils.core import Extension

nddf = Extension(name = 'nddf',
                   sources = ['exts/ndd.pyf','exts/gamma.f90','exts/quad.f90','exts/ndd.f90'],
#                   extra_f90_compile_args = ["-fopenmp"],
#                   extra_link_args = ["-lgomp"]
               )

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name              = 'ndd',
          description       = "Estimates of entropy and entropy-related quantities from discrete data",
          url               = '',
          author            = "Simone Marsili",
          author_email      = "simomarsili@gmail.com",
          license           = 'BSD 3 clause',
          #packages         = ['ndd'],
          py_modules        = ['ndd'],
          ext_modules       = [nddf]
          )
