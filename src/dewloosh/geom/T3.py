# -*- coding: utf-8 -*-
from dewloosh.geom.polygon import Triangle
from dewloosh.math.array import repeat
from numba import njit, prange
import numpy as np
from numpy import ndarray
__cache = True


@njit(nogil=True, cache=__cache)
def monoms_CST(pcoord: ndarray):
    r, s = pcoord[:2]
    return np.array([1, r, s])


@njit(nogil=True, cache=__cache)
def shp_CST(pcoord: ndarray):
    r, s = pcoord
    return np.array([1 - r - s, r, s])


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_CST(pcoord: np.ndarray):
    eye = np.eye(2, dtype=pcoord.dtype)
    shp = shp_CST(pcoord)
    res = np.zeros((2, 6), dtype=pcoord.dtype)
    for i in prange(3):
        res[:, i * 2: (i+1) * 2] = eye*shp[i]
    return res


@njit(nogil=True, cache=__cache)
def dshp_CST():
    return np.array([[-1, -1], [1, 0], [0, 1]])


class T3(Triangle):

    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([[0., 0.], [1., 0.], [0., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([[1/3, 1/3]])

    def shape_function_derivatives(self, coords=None, *args, **kwargs):
        if coords is None:
            coords = self.pointdata.x.to_numpy()
        if len(coords.shape) == 2:
            return repeat(dshp_CST(), coords.shape[0])
        else:
            return dshp_CST()
