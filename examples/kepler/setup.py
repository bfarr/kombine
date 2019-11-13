from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    name="rvfitting",
    version="0.0.1",
    cmdclass={"build_ext": build_ext},
    ext_modules=[Extension("kepler", ["kepler.pyx"], include_dirs=[np.get_include()])],
)
