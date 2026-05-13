from setuptools import setup, Extension
from numpy import get_include

ext_modules = [
    Extension(
        "cspy.flex.listmathfast",
        ["cspy/flex/listmathfast.pyx"],
        language="c++",
        include_dirs=[get_include()]
    ),
    Extension(
        "cspy.db.clustering_loops",
        ["cspy/db/clustering_loops.pyx"],
        language="c++",
        include_dirs=[get_include()]
    ),
    Extension(
        "cspy.sample.sobol",
        ["cspy/sample/sobol.pyx"],
        language="c++",
        include_dirs=[get_include()]
    ),
    Extension(
        "cspy.sample.lds",
        ["cspy/sample/lds.pyx"],
        language="c++",
        include_dirs=[get_include()]
    ),
    Extension(
        "cspy.ml.distance._cdtw",
        sources=["cspy/ml/distance/_cdtw.c"],
        language="c",
        include_dirs=[get_include()]
    ),
]

setup(
    ext_modules=ext_modules
)