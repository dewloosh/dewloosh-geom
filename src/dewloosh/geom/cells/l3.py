# -*- coding: utf-8 -*-
from dewloosh.geom.line import Line, QuadraticLine
from dewloosh.geom.utils import lengths_of_lines2
import numpy as np
from numba import njit, prange
import numpy as np
from numpy import ndarray
__cache = True


__all__ = ['L3']


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_det_bulk_1d(jac: ndarray):
    nE, nG = jac.shape[:2]
    res = np.zeros((nE, nG), dtype=jac.dtype)
    for iE in prange(nE):
        res[iE, :] = jac[iE, :, 0, 0]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_matrix_bulk_1d(dshp: ndarray, ecoords: ndarray):
    """
    Returns the Jacobian matrix for multiple (nE) elements, evaluated at
    multiple (nP) points.
    ---
    (nE, nP, 1, 1)
    
    Notes
    -----
    As long as the line is straight, it is a constant metric element,
    and 'dshp' is only required here to provide an output with a correct shape.
    """
    lengths = lengths_of_lines2(ecoords)
    nE = ecoords.shape[0]
    if len(dshp.shape) > 4:
        # variable metric element -> dshp (nE, nP, nNE, nDOF, ...)
        nP = dshp.shape[1]
    else:
        # constant metric element -> dshp (nP, nNE, nDOF, ...)
        nP = dshp.shape[0]
    res = np.zeros((nE, nP, 1, 1), dtype=dshp.dtype)
    for iE in prange(nE):
        res[iE, :, 0, 0] = lengths[iE] / 2
    return res


class L3(QuadraticLine):

    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([[-1., 0., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([0.])

    """def shape_function_values(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return shp2_bulk(coords)
        else:
            return shp2(coords)

    def shape_function_derivatives(self, coords, *args, **kwargs):
        if len(coords.shape) == 2:
            return dshp2_bulk(coords)
        else:
            return dshp2(coords)"""

    def volumes(self, *args, **kwargs):
        lengths = self.lengths(*args, **kwargs)
        if 'area' in self.fields:
            areas = self.area.to_numpy()
            return lengths * areas
        else:
            return lengths

    def jacobian_matrix(self, *args, dshp=None, **kwargs):
        assert dshp is not None
        ecoords = kwargs.get('ecoords', self.local_coordinates())
        return jacobian_matrix_bulk_1d(dshp, ecoords)

    def jacobian(self, *args, jac=None, **kwargs):
        return jacobian_det_bulk_1d(jac)