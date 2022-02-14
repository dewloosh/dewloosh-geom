# -*- coding: utf-8 -*-
from dewloosh.geom.line import Line
from dewloosh.geom.utils import lengths_of_lines2
import numpy as np
from numba import njit, prange
import numpy as np
from numpy import ndarray
__cache = True


__all__ = ['Line']


@njit(nogil=True, cache=__cache)
def monoms(r):
    return np.array([1, r])


@njit(nogil=True, cache=__cache)
def shp(r):
    return np.array([1-r, 1+r]) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def shp_bulk(pcoords: np.ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = shp(pcoords[iP])
    return res


@njit(nogil=True, cache=__cache)
def dshp(r):
    return np.array([-1, 1]) / 2


@njit(nogil=True, parallel=True, cache=__cache)
def dshp_bulk(pcoords: ndarray):
    nP = pcoords.shape[0]
    res = np.zeros((nP, 2), dtype=pcoords.dtype)
    for iP in prange(nP):
        res[iP, :] = dshp(pcoords[iP])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_det_bulk_1d(jac: ndarray):
    nE, nG = jac.shape[:2]
    res = np.zeros((nE, nG), dtype=jac.dtype)
    for iE in prange(nE):
        res[iE, :] = jac[iE, :, 0, 0]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_matrix_bulk_1d(dshp: ndarray, ecoords: ndarray):
    lengths = lengths_of_lines2(ecoords)
    nE = ecoords.shape[0]
    nG = dshp.shape[0]
    res = np.zeros((nE, nG, 1, 1), dtype=dshp.dtype)
    for iE in prange(nE):
        res[iE, :, 0, 0] = lengths[iE] / 2
    return res


class L2(Line):
    
    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([[-1., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([0.])
                        
    def shape_function_values(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return shp_bulk(coords)
        else:
            return shp(coords)

    def shape_function_derivatives(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return dshp_bulk(coords)
        else:
            return dshp(coords)
        
    def volumes(self, *args, **kwargs):
        lengths = self.lengths(*args, **kwargs)
        if 'area' in self.fields:
            areas = self.area.to_numpy()
            return lengths * areas
        else:
            return lengths
        