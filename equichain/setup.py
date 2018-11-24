from distutils.core import setup,Extension
from Cython.Build import cythonize

ext_modules = [
        Extension('cython_fns',
            sources=['cython_fns.pyx'],
            ),
        Extension('magmaconv_cython',
            sources=['magmaconv_cython.pyx'],
            language="c++",
            extra_compile_args=["-std=c++11"]
            )
                ]

setup(
    ext_modules = cythonize(ext_modules)
)
