import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        'core_vectordb',
        [
            'core/bindings/PybindWrapper.cpp',
            'core/src/Distance.cpp',
            'core/src/BruteForce.cpp',
            'core/src/KDTree.cpp',
            'core/src/HNSW.cpp',
        ],
        include_dirs=[
            'core/include',
            pybind11.get_include(),
            pybind11.get_include(user=True)
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]

setup(
    name='core_vectordb',
    version='0.1.0',
    description='C++ Core VectorDB with pybind11',
    ext_modules=ext_modules,
)
