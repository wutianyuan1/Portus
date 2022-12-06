

from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "rpma_server",
        ["interface.cpp", "../chksystem.cpp", "../object.cpp", "../pool.cpp"],
        include_dirs=["../", "./"],
        cxx_std="17",
        extra_compile_args=["-O3", "-Wall", "-mclwb"],
    ),
]

setup(name='rpma_server', ext_modules=ext_modules)