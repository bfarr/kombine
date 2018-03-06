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
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
    ],
)
