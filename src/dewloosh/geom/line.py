# -*- coding: utf-8 -*-
import vtk
from dewloosh.geom.utils import lengths_of_lines
from dewloosh.geom.cell import PolyCell1d
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


class Line(PolyCell1d):
    
    NNODE = 2
    vtkCellType = vtk.VTK_LINE
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @classmethod
    def lcoords(cls, *args, **kwargs):
        return np.array([[-1., 1.]])

    @classmethod
    def lcenter(cls, *args, **kwargs):
        return np.array([0.])
        
    def lengths(self, *args, coords=None, topo=None, **kwargs):
        coords = self.pointdata.x.to_numpy() if coords is None else coords
        topo = self.nodes.to_numpy() if topo is None else topo
        return lengths_of_lines(coords, topo)
    
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
        
    def volume(self, *args, **kwargs):
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs):
        lengths = self.lengths(*args, **kwargs)
        if 'area' in self.fields:
            areas = self.area.to_numpy()
            return lengths * areas
        else:
            return lengths