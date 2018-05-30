#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

import build

this_file = os.path.dirname(__file__)

setup(
    name="torchain",
    version="0.4",
    description="Kaldi's chain binding for PyTorch",
    url="http://kishin-gitlab.cslab.kecl.ntt.co.jp/karita/torchain",
    author="Shigeki Karita",
    author_email="karita.shigeki@lab.ntt.co.jp",
    # Require cffi.
    install_requires=["cffi>=1.0.0", "torch>=0.3.1"],
    setup_requires=["cffi>=1.0.0", "torch>=0.3.1"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)
