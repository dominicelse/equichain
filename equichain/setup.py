from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
        Extension('magmaconv_cython',
            sources=['magmaconv_cython.pyx'],
            language="c++",
            extra_compile_args=["-std=c++11"],
            include_dirs=[numpy.get_include()]
            ),
        Extension('sageinit',
            sources=['sageinit.pyx'],
            language="c",
            include_dirs=[numpy.get_include()]
            )
        ]

setup(
    ext_modules = cythonize(ext_modules)
)
