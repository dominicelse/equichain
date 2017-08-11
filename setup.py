from distutils.core import setup,Extension
from Cython.Build import cythonize

ext_modules = [
        Extension('cython_fns',
            sources=['cython_fns.pyx']
            )
            ]

setup(
    ext_modules = cythonize(ext_modules)
)
