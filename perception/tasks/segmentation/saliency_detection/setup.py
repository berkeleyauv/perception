from distutils.core import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
# define an extension that will be cythonized and compiled
ext = Extension(name="saliency_mbd", sources=["saliency_mbd.pyx"])
ext2 = Extension(name="prange_saliency_mbd", sources=["prange_saliency_mbd.pyx"])
setup(ext_modules=[ext2,ext], include_dirs=[numpy.get_include()],
    cmdclass = {'build_ext': build_ext})
 