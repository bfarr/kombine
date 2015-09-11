#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import textwrap

import numpy as np
import scipy

from .interruptible_pool import disable_openblas_threading as ip_disable_threading

def bad_blas_msg(pkg_name):
    """A warning message to give the user"""
    return textwrap.dedent(
        """
        {0} linked against 'Accelerate.framework', which  doesn't play nicely
        with 'multiprocessing'.

        Building {0} against OpenBLAS can avoid this, e.g.:

            brew tap homebrew/python
            brew update && brew upgrade

            brew install openblas

            brew install {0} --with-openblas

        To maintain stability, multiprocessing won't be used.
        """.format(pkg_name))

def get_config_info(pkg):
    """Extract config info from Scipy or Numpy"""
    try:
        config_info = str([value for key, value in
                           pkg.__config__.__dict__.iteritems()
                           if key.endswith("_info")]).lower()
    except AttributeError:
        config_info = str([value for key, value in
                           pkg.__config__.__dict__.items()
                           if key.endswith("_info")]).lower()
    return config_info

def mp_safe_blas():
    """Check if BLAS implementation is known to not play nicely with :mod:`multiprocessing`."""
    safe_blas = True

    np_config_info = get_config_info(np)
    scipy_config_info = get_config_info(scipy)

    if "accelerate" in np_config_info or "veclib" in np_config_info:
        warnings.warn(bad_blas_msg('numpy'))
        safe_blas = False

    if "accelerate" in scipy_config_info or "veclib" in scipy_config_info:
        warnings.warn(bad_blas_msg('scipy'))
        safe_blas = False

    return safe_blas

def disable_openblas_threading():
    """If openblas is linked, disable threading to avoid extra overhead"""
    np_config_info = get_config_info(np)
    scipy_config_info = get_config_info(scipy)

    if "openblas" in np_config_info or "openblas" in scipy_config_info:
        ip_disable_threading()
