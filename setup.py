#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

setup(
    name='kombine',
    version='1.0',
    description='An embarrassingly parallel, kernel-density-based\
                 ensemble sampler',
    author='Ben Farr',
    author_email='farr@uchicago.edu',
    url='https://github.com/bfarr/kombine',
    include_package_data=True,
    packages=['kombine'],
    install_requires=['numpy', 'scipy'],
)
