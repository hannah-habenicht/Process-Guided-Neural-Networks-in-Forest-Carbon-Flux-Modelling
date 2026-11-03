# setup.py
from setuptools import setup, Extension
import pybind11
import numpy

ext = Extension(
    name="preles_c",
    sources=[
        "bindings.cpp",      # the pybind11 bridge
        "preles.c",
        "gpp.c",
        "water.c",
        "initruns.c",
    ],
    include_dirs=[
        pybind11.get_include(),
        numpy.get_include(),
        ".",                 # for prelesglobals.h / preles_c_api.h
    ],
    extra_compile_args=["-O3", "-fPIC"],   # add "-fopenmp" if your C uses OpenMP
    # extra_link_args=["-fopenmp"],        # if OpenMP used with GCC/Clang
)

setup(
    name="preles_c",
    version="0.1.0",
    ext_modules=[ext],
)
