"""Variogram and Covariance Functions"""
from typing import Union

import numpy as np
from numba import jit
from numpy import ndarray


@jit(nopython=True)
def exponential_covariance_function(
    h: Union[float, np.array], c0: float, c1: float, a: float
) -> ndarray:
    """
    Exponential covariance function
    See exponential_variogram_function for argument description
    :return: exponential covariance value for distance
    """
    return np.where(h == 0, c0 + c1, c1 * (np.exp(-h / a)))


@jit(nopython=True)
def exponential_variogram_function(
    h: Union[float, np.array], c0: float, c1: float, a: float
) -> float:
    """
    Exponential variogram function
    :param c0: nugget effect, which provides discontinuity at the origin
    :param c1: c0 + c1 = sill, which is the variogram value for large distances. It is also the variance
    :param h: distance between coordinates
    :param a: range, which provides a distance beyond which the variogram or covariance value remains constant
    :return: exponential variogram value for distance
    """
    return c0 + c1 * (1 - np.exp(-h / a))
