# -*- coding: utf-8 -*-
# !TODO  handle decimation in all transformations, template : T6_to_T3
# !TODO  correct all transformations, template : Q4_to_T3
import numpy as np
from numba import njit, prange
from dewloosh.geom.tri.triutils import edges_tri
from dewloosh.geom.utils import cell_coords_bulk, \
    detach_mesh_bulk as detach_mesh
from dewloosh.geom.topo.topodata import edges_Q4, edges_H8, faces_H8
from dewloosh.geom.topo import unique_topo_data
from typing import Union, Sequence
from collections import Iterable
from numpy import ndarray
from concurrent.futures import ThreadPoolExecutor
__cache = True


__all__ = [
    'T3_to_T6', 'T6_to_T3',
    'Q4_to_Q8',
    'Q4_to_Q9', 'Q9_to_Q4',
    'Q9_to_T6',
    'Q4_to_T3',
    'H8_to_H27'
]


DataLike = Union[ndarray, Sequence[ndarray]]


def transform_topo(topo: ndarray, path: ndarray, data: ndarray = None,
                   *args, MT=True, max_workers=4, **kwargs):
    nD = len(path.shape)
    if nD == 1:
        path = path.reshape(1, len(path))
    assert nD <= 2, "Path must be 1 or 2 dimensional."
    if data is None:
        return _transform_topo_(topo, path)
    else:
        if isinstance(data, ndarray):
            data = transform_topo_data(topo, data, path)
            return _transform_topo_(topo, path), data
        elif isinstance(data, Iterable):
            def foo(d): return transform_topo_data(topo, d, path)
            if MT:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    dmap = executor.map(foo, data)
            else:
                dmap = map(foo, data)
            return _transform_topo_(topo, path), list(dmap)


def transform_topo_data(topo: ndarray, data: ndarray, path: ndarray):
    assert topo.shape[0] == data.shape[0]
    if data.shape[:2] == topo.shape[:2]:
        res = repeat_cell_nodal_data(data, path)
    else:
        res = np.repeat(data, path.shape[0], axis=0)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _transform_topo_(topo: ndarray, path: ndarray):
    nE = len(topo)
    nSub, nSubN = path.shape
    res = np.zeros((nSub * nE, nSubN), dtype=topo.dtype)
    for iE in prange(nE):
        c = iE * nSub
        for jE in prange(nSubN):
            for kE in prange(nSub):
                res[c + kE, jE] = topo[iE, path[kE, jE]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def repeat_cell_nodal_data(edata: ndarray, path: ndarray):
    nSub, nSubN = path.shape
    nE = edata.shape[0]
    res = np.zeros((nSub*nE, nSubN) + edata.shape[2:], dtype=edata.dtype)
    for i in prange(nE):
        ii = nSub*i
        for j in prange(nSub):
            jj = ii+j
            for k in prange(nSubN):
                res[jj, k] = edata[i, path[j, k]]
    return res


def T3_to_T6(coords: ndarray, topo: ndarray):
    nP = len(coords)
    edges, edgeIDs = unique_topo_data(edges_tri(topo))
    new_coords = np.mean(coords[edges], axis=1)
    new_topo = edgeIDs + nP
    topo = np.hstack((topo, new_topo))
    coords = np.vstack((coords, new_coords))
    return coords, topo


def T6_to_T3(coords: ndarray, topo: ndarray, *args,
             path: ndarray = None, edata=None, decimate=True,
             return_inverse=False, **kwargs):
    if edata is not None:
        assert isinstance(edata, ndarray), \
            "If 'edata' is provided, it must be a numpy array!"
    if decimate:
        if path is None:
            path = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [5, 3, 4]],
                            dtype=topo.dtype)
        if edata is not None:
            data_T3 = np.repeat(edata, 4, axis=0)
            return T6_to_T3_h(coords, topo, path) + (data_T3,)
        else:
            return T6_to_T3_h(coords, topo, path)
    else:
        if edata is not None:
            return detach_mesh(coords, topo[:, [0, 1, 2]]) + (edata,)
        else:
            return detach_mesh(coords, topo[:, [0, 1, 2]])


@njit(nogil=True, parallel=True, cache=__cache)
def T6_to_T3_h(coords: ndarray, topo: ndarray, path: ndarray):
    nE = len(topo)
    topoT3 = np.zeros((4 * nE, 3), dtype=topo.dtype)
    for iE in prange(nE):
        n = iE * 4
        for j in prange(4):
            for k in prange(3):
                topoT3[n + j, k] = topo[iE, path[j, k]]
    return coords, topoT3


def Q4_to_Q8(coords: ndarray, topo: ndarray):
    nP = len(coords)
    edges, edgeIDs = unique_topo_data(edges_Q4(topo))
    new_coords = np.mean(coords[edges], axis=1)
    new_topo = edgeIDs + nP
    topo = np.hstack((topo, new_topo))
    coords = np.vstack((coords, new_coords))
    return coords, topo


def Q4_to_Q9(coords: ndarray, topo: ndarray):
    nP, nE = len(coords), len(topo)
    # new nodes on the edges
    edges, edgeIDs = unique_topo_data(edges_Q4(topo))
    coords_e = np.mean(coords[edges], axis=1)
    topo_e = edgeIDs + nP
    nP += len(coords_e)
    # new coords at element centers
    ecoords = cell_coords_bulk(coords, topo)
    coords_c = np.mean(ecoords, axis=1)
    topo_c = np.arange(nE) + nP
    # assemble
    topo_res = np.hstack((topo, topo_e, topo_c))
    coords_res = np.vstack((coords, coords_e, coords_c))
    return coords_res, topo_res


def Q9_to_Q4(coords: ndarray, topo: ndarray, data: DataLike = None,
             *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[1] == 4
    else:
        if path is None:
            path = np.array([[0, 4, 8, 7], [4, 1, 5, 8],
                             [8, 5, 2, 6], [7, 8, 6, 3]],
                            dtype=topo.dtype)
        elif isinstance(path, str):
            if path == 'grid':
                path = np.array([[0, 3, 4, 1], [3, 6, 7, 4],
                                 [4, 7, 8, 5], [1, 4, 5, 2]],
                                dtype=topo.dtype)
    if data is None:
        return coords, + transform_topo(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topo(topo, path, data, *args, **kwargs)


def Q9_to_T6(coords: ndarray, topo: ndarray, path: ndarray = None):
    if path is None:
        path = np.array([[0, 8, 2, 4, 5, 1], [0, 6, 8, 3, 7, 4]],
                        dtype=topo.dtype)
    return _Q9_to_T6(coords, topo, path)


@njit(nogil=True, parallel=True, cache=__cache)
def _Q9_to_T6(coords: ndarray, topo: ndarray, path: ndarray):
    nE = len(topo)
    topoT6 = np.zeros((2 * nE, 6), dtype=topo.dtype)
    for iE in prange(nE):
        c = iE * 2
        for jE in prange(6):
            topoT6[c, jE] = topo[iE, path[0, jE]]
            topoT6[c + 1, jE] = topo[iE, path[1, jE]]
    return coords, topoT6


def Q4_to_T3(coords: ndarray, topo: ndarray, data: DataLike = None,
             *args, path: ndarray = None, **kwargs):
    if isinstance(path, ndarray):
        assert path.shape[1] == 3
    else:
        if path is None:
            path = np.array([[0, 1, 2], [0, 2, 3]], dtype=topo.dtype)
        elif isinstance(path, str):
            if path == 'grid':
                path = np.array([[0, 2, 3], [0, 3, 1]], dtype=topo.dtype)
    if data is None:
        return coords, + transform_topo(topo, path, *args, **kwargs)
    else:
        return (coords,) + transform_topo(topo, path, data, *args, **kwargs)


def H8_to_H27(coords: ndarray, topo: ndarray):
    nP, nE = len(coords), len(topo)
    ecoords = cell_coords_bulk(coords, topo)
    # new nodes on the edges
    edges, edgeIDs = unique_topo_data(edges_H8(topo))
    coords_e = np.mean(coords[edges], axis=1)
    topo_e = edgeIDs + nP
    nP += len(coords_e)
    # new nodes on face centers
    faces, faceIDs = unique_topo_data(faces_H8(topo))
    coords_f = np.mean(coords[faces], axis=1)
    topo_f = faceIDs + nP
    nP += len(coords_f)
    # register new nodes in the cell centers
    coords_c = np.mean(ecoords, axis=1)
    topo_c = np.reshape(np.arange(nE) + nP, (nE, 1))
    # assemble
    topo_res = np.hstack((topo, topo_e, topo_f, topo_c))
    coords_res = np.vstack((coords, coords_e, coords_f, coords_c))
    return coords_res, topo_res
