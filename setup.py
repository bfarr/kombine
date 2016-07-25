#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import re

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

version_re = re.compile("__version__ = \"(.*?)\"")
with open(path.join(path.dirname(path.abspath(__file__)), "kombine", "__init__.py")) as inp:
    r = inp.read()
version = version_re.findall(r)[0]

setup(
    name='kombine',
    version=version,
    description='An embarrassingly parallel, kernel-density-based\
                 ensemble sampler',
    author='Ben Farr',
    author_email='farr@uchicago.edu',
    url='https://github.com/bfarr/kombine',
    include_package_data=True,
    packages=['kombine'],
)
