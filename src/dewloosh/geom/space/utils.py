# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
from dewloosh.core.squeeze import squeeze
from dewloosh.math.linalg import normalize
from dewloosh.math.array import atleast2d
from dewloosh.geom.utils import center_of_points, \
    cell_center, cell_coords
__cache = True


@njit(nogil=True, cache=__cache)
def frame_of_plane(coords: ndarray):
    tr = np.zeros((3, 3), dtype=coords.dtype)
    center = center_of_points(coords)
    tr[:, 0] = normalize(coords[0] - center)
    tr[:, 1] = normalize(coords[np.int(len(coords)/2)] - center)
    tr[:, 2] = np.cross(tr[:, 0], tr[:, 1])
    tr[:, 1] = np.cross(tr[:, 2], tr[:, 0])
    return center, tr


@njit(nogil=True, parallel=True, cache=__cache)
def frames_of_surfaces(coords: ndarray, topo: ndarray):
    """
    Returns the coordinates of the axes forming the local 
    coordinate systems of the surfaces.

    Parameters
    ----------
    coords : numpy.ndarray
        2d coordinate array

    topo : numpy.ndarray
        2d point-based topology array

    Returns:
    --------        
    numpy.ndarray
        3d array of 3x3 transformation matrices
    """
    nE, nNE = topo.shape
    nNE -= 1
    tr = np.zeros((nE, 3, 3), dtype=coords.dtype)
    for iE in prange(nE):
        tr[iE, 0, :] = normalize(coords[topo[iE, 1]] - coords[topo[iE, 0]])
        tr[iE, 1, :] = normalize(coords[topo[iE, nNE]] - coords[topo[iE, 0]])
        tr[iE, 2, :] = np.cross(tr[iE, 0, :], tr[iE, 1, :])
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
    return tr


@njit(nogil=True, parallel=True, cache=__cache)
def tr_cell_glob_to_loc_bulk(coords: np.ndarray, topo: np.ndarray):
    """
    Returns the coordinates of the cells in their local coordinate
    system, the coordinates of their centers and the coordinates of the 
    axes forming their local coordinate system. The local coordinate systems 
    are located at the centers of the cells.

    Parameters
    ----------
    coords : numpy.ndarray
        2d coordinate array

    topo : numpy.ndarray
        2d point-based topology array

    Returns:
    --------
    numpy.ndarray
        2d coordinate array of local coordinates

    numpy.ndarray
        2d array of cell centers

    numpy.ndarray
        3d array of 3x3 transformation matrices
    """
    nE, nNE = topo.shape
    tr = np.zeros((nE, 3, 3), dtype=coords.dtype)
    res = np.zeros((nE, nNE, 2), dtype=coords.dtype)
    centers = np.zeros((nE, 3), dtype=coords.dtype)
    for iE in prange(nE):
        centers[iE] = cell_center(cell_coords(coords, topo[iE]))
        tr[iE, 0, :] = normalize(coords[topo[iE, 1]] - coords[topo[iE, 0]])
        tr[iE, 1, :] = normalize(coords[topo[iE, nNE-1]] - coords[topo[iE, 0]])
        tr[iE, 2, :] = np.cross(tr[iE, 0, :], tr[iE, 1, :])
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
        for jN in prange(nNE):
            vj = coords[topo[iE, jN]] - centers[iE]
            res[iE, jN, 0] = np.dot(tr[iE, 0, :], vj)
            res[iE, jN, 1] = np.dot(tr[iE, 1, :], vj)
    return res, centers, tr


@njit(nogil=True, parallel=True, cache=__cache)
def _frames_of_lines_auto(coords: ndarray, topo: ndarray):
    nE, nNE = topo.shape
    nNE -= 1
    k = np.array([0, 0, 1], dtype=coords.dtype)
    tr = np.zeros((nE, 3, 3), dtype=coords.dtype)
    for iE in prange(nE):
        tr[iE, 0, :] = normalize(coords[topo[iE, nNE]] - coords[topo[iE, 0]])
        tr[iE, 2, :] = normalize(k - tr[iE, 0, :] * np.dot(tr[iE, 0, :], k))
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
    return tr


@njit(nogil=True, parallel=True, cache=__cache)
def _frames_of_lines_ref(coords: ndarray, topo: ndarray, refZ: ndarray):
    nE, nNE = topo.shape
    nNE -= 1
    tr = np.zeros((nE, 3, 3), dtype=coords.dtype)
    for iE in prange(nE):
        tr[iE, 0, :] = normalize(coords[topo[iE, nNE]] - coords[topo[iE, 0]])
        k = refZ[iE] - coords[topo[iE, 0]]
        tr[iE, 2, :] = normalize(k - tr[iE, 0, :] * np.dot(tr[iE, 0, :], k))
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
    return tr


@squeeze(True)
def frames_of_lines(coords: ndarray, topo: ndarray, refZ: ndarray=None):
    """
    Returns coordinate frames of line elements defined by a coordinate array
    and a topology array. The cross-sections of the line elements are
    in the local y-z plane. The direction of the local z axis can be set by 
    providing reference points in the local x-z planes of the lines. If there
    are no references provided, local z axes lean towards global z. 
    Other properties are determined in a way, so that x-y-z form a 
    right-handed orthonormal basis.
    
    Parameters
    ----------
    coords : numpy.ndarray
        2d coordinate array

    topo : numpy.ndarray
        2d point-based topology array
        
    refZ : numpy.ndarray, Optional
        1d or 2d float array of reference points. If it is 2d, it must
        contain values for all lines defined by `topo`. 
        Default is None.

    Returns:
    --------        
    numpy.ndarray
        3d array of 3x3 transformation matrices
        
    """
    topo = atleast2d(topo)
    if isinstance(refZ, ndarray):
        if len(topo.shape) == 2 and len(refZ.shape) == 1:
            _refZ = np.zeros((topo.shape[0], 3))
            _refZ[:] = refZ
        else:
            _refZ = refZ
        return _frames_of_lines_ref(coords, topo, _refZ)
    else:
        return _frames_of_lines_auto(coords, topo)



