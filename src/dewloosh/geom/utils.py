# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from numba import njit, prange
from numba.typed import Dict as nbDict
from numba import types as nbtypes
from dewloosh.math.array import matrixform
from dewloosh.geom.topo import remap_topo
from dewloosh.math.linalg.sparse import JaggedArray
from dewloosh.math.linalg.sparse.csr import csr_matrix
__cache = True


nbint64 = nbtypes.int64
nbint64A = nbint64[:]
nbfloat64A = nbtypes.float64[:]


def cells_around(centers: np.ndarray, r_max: float, *args,
                 frmt='dict', MT=True, n_max:int=10, **kwargs):
    if MT:
        data, widths = _cells_around_MT_(centers, r_max, n_max)    
    else:
        raise NotImplementedError
    if frmt == 'dict':
        return _cells_data_to_dict(data, widths)
    elif frmt == 'jagged':
        return _cells_data_to_jagged(data, widths)
    elif frmt == 'csr':
        d = _cells_data_to_dict(data, widths)
        data, inds, indptr, shp = _dict_to_spdata(d, widths)
        return csr_matrix(data=data, indices=inds, 
                          indptr=indptr, shape=shp)
    raise RuntimeError("Unhandled case!")


@njit(nogil=True, cache=__cache)
def _dict_to_spdata(d: dict, widths: np.ndarray):
    N = int(np.sum(widths))
    nE = len(widths)
    data = np.zeros(N, dtype=np.int64)
    inds = np.zeros_like(data)
    indptr = np.zeros(nE+1, dtype=np.int64)
    _c = 0
    wmax = 0
    for i in range(len(d)):
        w = widths[i]
        if w > wmax:
            wmax = w
        c_ = _c + w
        data[_c: c_] = d[i]
        inds[_c: c_] = np.arange(w)
        indptr[i+1] = c_
        _c = c_
    return data, inds, indptr, (nE, wmax)


@njit(nogil=True, cache=__cache)
def _jagged_to_spdata(ja: JaggedArray):
    widths = ja.widths()
    N = int(np.sum(widths))
    nE = len(widths)
    data = np.zeros(N, dtype=np.int64)
    inds = np.zeros_like(data)
    indptr = np.zeros(nE+1, dtype=np.int64)
    _c = 0
    wmax = 0
    for i in range(len(ja)):
        w = widths[i]
        if w > wmax:
            wmax = w
        c_ = _c + w
        data[_c: c_] = ja[i]
        inds[_c: c_] = np.arange(w)
        indptr[i+1] = c_
        _c = c_
    return data, inds, indptr, (nE, wmax)


@njit(nogil=True, fastmath=True, cache=__cache)
def _cells_data_to_dict(data: np.ndarray, widths: np.ndarray) -> nbDict:
    dres = dict()
    nE = len(widths)
    for iE in range(nE):
        dres[iE] = data[iE, :widths[iE]]
    return dres


@njit(nogil=True, parallel=True, cache=__cache)
def _flatten_jagged_data(data, widths) -> ndarray:
    nE = len(widths)
    inds = np.zeros(nE + 1, dtype=widths.dtype)
    inds[1:] = np.cumsum(widths)
    res = np.zeros(np.sum(widths))
    for i in prange(nE):
        res[inds[i] : inds[i+1]] = data[i, :widths[i]]
    return res


def _cells_data_to_jagged(data, widths):
    data = _flatten_jagged_data(data, widths)
    return JaggedArray(data, cuts=widths)


@njit(nogil=True, cache=__cache)
def _cells_around_(centers: np.ndarray, r_max: float):
    res = nbDict.empty(
        key_type=nbint64,
        value_type=nbint64A,
    )
    nE = len(centers)
    normsbuf = np.zeros(nE, dtype=centers.dtype)
    widths = np.zeros(nE, dtype=np.int64)
    for iE in range(nE):
        normsbuf[:] = norms(centers - centers[iE])
        res[iE] = np.where(normsbuf <= r_max)[0]
        widths[iE] = len(res[iE])
    return res, widths


@njit(nogil=True, parallel=True, cache=__cache)
def _cells_around_MT_(centers: np.ndarray, r_max: float, n_max: int=10):
    nE = len(centers)
    res = np.zeros((nE, n_max), dtype=np.int64)
    widths = np.zeros(nE, dtype=np.int64)
    for iE in prange(nE):
        inds = np.where(norms(centers - centers[iE]) <= r_max)[0]
        if inds.shape[0] <= n_max:
            res[iE, :inds.shape[0]] = inds
        else:
            res[iE, :] = inds[:n_max]
        widths[iE] = len(res[iE])
    return res, widths


def index_of_closest_point(coords: ndarray, target: ndarray) -> int:
    """
    Returs the index of the closes point to a target.
    
    Parameters
    ----------
    coords : (nP, nD) numpy.ndarray
        2d float array of vertex coordinates.
            nP : number of points in the model
            nD : number of dimensions of the model space
            
    target : numpy.ndarray
        Coordinate array of the target point.

    Returns
    -------
    int
        The index of `coords`, for which the distance from
        `target` is minimal.
    """
    assert coords.shape[1] == target.shape[0], \
        "The dimensions of `coords` and `target` are not compatible."
    return np.argmin(norm(coords - target, axis=1))


def points_of_cells(coords: ndarray, topo: ndarray, *args, 
                    local_axes=None, centralize=True, **kwargs) -> ndarray:
    """
    Returns coordinates of the cells from a pointset and a topology.
    If coordinate frames are provided, the coorindates are returned with
    respect to those frames.
    
    Parameters
    ----------
    coords : (nP, nD) numpy.ndarray
        2d float array of vertex coordinates.
            nP : number of points in the model
            nD : number of dimensions of the model space
            
    topo : (nE, nNE) numpy.ndarray
        A 2D array of vertex indices. The i-th row contains the vertex indices
        of the i-th element.
            nE : number of elements
            nNE : number of nodes per element
            
    local_axes : (..., 3, 3) numpy.ndarray
        Reference frames. A single 3x3 numpy array or matrices for all elements
        in 'topo' must be provided.
            
    centralize : bool
        If True, and 'frame' is not None, the local coordinates are returned
        with respect to the geometric center of each element.
    
    Notes
    -----
    It is assumed that all entries in 'coords' are coordinates of
    points in the same frame.
    """
    if local_axes is not None:
        return cells_coords_tr(cells_coords(coords, topo), 
                               local_axes, centralize=centralize)
    else:
        return cells_coords(coords, topo)
    
    
@njit(nogil=True, parallel=True, cache=__cache)
def cells_coords_tr(ecoords: ndarray, local_axes: ndarray, 
                    centralize=True) -> ndarray:
    nE, nNE, _ = ecoords.shape
    res = np.zeros_like(ecoords)
    for i in prange(nE):
        if centralize:
            cc = cell_center(ecoords[i])
            ecoords[i, :, 0] -= cc[0]
            ecoords[i, :, 1] -= cc[1]
            ecoords[i, :, 2] -= cc[2]
        dcm = local_axes[i]
        for j in prange(nNE):
            res[i, j, :] = dcm @ ecoords[i, j, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def cells_coords(coords: ndarray, topo: ndarray) -> ndarray:
    """Returns coordinates of multiple cells.

    Returns coordinates of cells from a coordinate base array and
    a topology array.

    Parameters
    ----------
    coords : (nP, nD) numpy.ndarray
        2d float array of all vertex coordinates of an assembly.
            nP : number of points in the model
            nD : number of dimensions of the model space
            
    topo : (nE, nNE) numpy.ndarray
        A 2D array of vertex indices. The i-th row contains the vertex indices
        of the i-th element.
            nE : number of elements
            nNE : number of nodes per element

    Returns
    -------
    (nE, nNE, nD) numpy.ndarray
        Coordinates for all nodes of all cells according to the
        argument 'topo'.

    Notes
    -----
    The array 'coords' must be fully populated up to the maximum index
    in 'topo'. (len(coords) >= (topo.max() + 1))

    Examples
    --------
    Typical usage:

    >>> coords = assembly.coords()
    >>> topo = assembly.topology()
    >>> print(len(coords) >= (topo.max() + 1))
    True
    >>> print(cell_coords_bulk(coords, topo))
    ...
    """
    nE, nNE = topo.shape
    res = np.zeros((nE, nNE, coords.shape[1]), dtype=coords.dtype)
    for iE in prange(nE):
        for iNE in prange(nNE):
            res[iE, iNE] = coords[topo[iE, iNE]]
    return res


cell_coords_bulk = cells_coords


@njit(nogil=True, parallel=True, cache=__cache)
def cell_coords(coords: ndarray, topo: ndarray) -> ndarray:
    """Returns coordinates of multiple cells.

    Returns coordinates of cells from a coordinate base array and
    a topology array.

    Parameters
    ----------
    coords : (nP, nD) numpy.ndarray
        Array of all vertex coordinates of an assembly.
            nP : number of points in the model
            nD : number of dimensions of the model space
    topo : (nNE) numpy.ndarray
        1D array of vertex indices.
            nNE : number of nodes per element

    Returns
    -------
    (nNE, nD) numpy.ndarray
        Coordinates for all nodes of all cells according to the
        argument 'topo'.

    Notes
    -----
    The array 'coords' must be fully populated up to the maximum index
    in 'topo'. (len(coords) >= (topo.max() + 1))

    Examples
    --------
    Typical usage:

    >>> coords = assembly.coords()
    >>> topo = assembly.topology()
    >>> print(len(coords) >= (topo.max() + 1))
    True
    >>> print(cell_coords(coords, topo[0]))
    ...
    """
    nNE = len(topo)
    res = np.zeros((nNE, coords.shape[1]), dtype=coords.dtype)
    for iNE in prange(nNE):
        res[iNE] = coords[topo[iNE]]
    return res


@njit(nogil=True, cache=__cache)
def cell_center_2d(ecoords: np.ndarray):
    """Returns the center of a 2d cell.

    Parameters
    ----------
    ecoords : numpy.ndarray
        2d coordinate array of the element. The array has as many rows,
        as the number of nodes of the cell, and two columns.

    Returns:
    --------
    numpy.ndarray
        1d coordinate array
    """
    return np.array([np.mean(ecoords[:, 0]), np.mean(ecoords[:, 1])],
                    dtype=ecoords.dtype)


@njit(nogil=True, cache=__cache)
def cell_center(coords: np.ndarray):
    """Returns the center of a general cell.

    Parameters
    ----------
    ecoords : numpy.ndarray
        2d coordinate array of the element. The array has as many rows,
        as the number of nodes of the cell, and three columns.

    Returns:
    --------
    numpy.ndarray
        1d coordinate array
    """
    return np.array([np.mean(coords[:, 0]), np.mean(coords[:, 1]),
                     np.mean(coords[:, 2])], dtype=coords.dtype)


def cell_center_bulk(coords: ndarray, topo: ndarray) -> ndarray:
    """Returns coordinates of the centers of the provided cells.

    Parameters
    ----------
    coords : numpy.ndarray
        2d coordinate array

    topo : numpy.ndarray
        2d point-based topology array

    Returns:
    --------
    numpy.ndarray
        2d coordinate array
    """
    return np.mean(cell_coords_bulk(coords, topo), axis=1)


@njit(nogil=True, parallel=True, cache=__cache)
def nodal_distribution_factors(topo: ndarray, volumes: ndarray):
    """The j-th factor of the i-th row is the contribution of
    element i to the j-th node. Assumes a regular topology."""
    factors = np.zeros(topo.shape, dtype=volumes.dtype)
    nodal_volumes = np.zeros(topo.max() + 1, dtype=volumes.dtype)
    for iE in range(topo.shape[0]):
        nodal_volumes[topo[iE]] += volumes[iE]
    for iE in prange(topo.shape[0]):
        for jNE in prange(topo.shape[1]):
            factors[iE, jNE] = volumes[iE] / nodal_volumes[topo[iE, jNE]]
    return factors


@njit(nogil=True, parallel=True, cache=__cache)
def distribute_nodal_data(data: ndarray, topo: ndarray, ndf: ndarray):
    nE, nNE = topo.shape
    res = np.zeros((nE, nNE, data.shape[1]))
    for iE in prange(nE):
        for jNE in prange(nNE):
            res[iE, jNE] = data[topo[iE, jNE]] * ndf[iE, jNE]
    return res


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def collect_nodal_data(celldata: ndarray, topo: ndarray, N: int):
    nE, nNE = topo.shape
    res = np.zeros((N, celldata.shape[2]), dtype=celldata.dtype)
    for iE in prange(nE):
        for jNE in prange(nNE):
            res[topo[iE, jNE]] += celldata[iE, jNE]
    return res


@njit(nogil=True, cache=__cache)
def inds_to_invmap_as_dict(inds: np.ndarray):
    """
    Returns a mapping that maps global indices to local ones.

    Parameters
    ----------
    inds : numpy.ndarray
        An array of global indices.

    Returns
    -------
    dict
        Mapping from global to local.
    """
    res = dict()
    for i in range(len(inds)):
        res[inds[i]] = i
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inds_to_invmap_as_array(inds: np.ndarray):
    """
    Returns a mapping that maps global indices to local ones
    as an array.

    Parameters
    ----------
    inds : numpy.ndarray
        An array of global indices.

    Returns
    -------
    numpy.ndarray
        Mapping from global to local.
    """
    res = np.zeros(inds.max() + 1, dtype=inds.dtype)
    for i in prange(len(inds)):
        res[inds[i]] = i
    return res


@njit(nogil=True, cache=__cache)
def detach_mesh_bulk(coords: ndarray, topo: ndarray):
    """
    Given a topology array and the coordinate array it refers to, 
    the function returns the coordinate array of the points involved 
    in the topology, and a new topology array, with indices referencing 
    the new coordinate array. 
    """
    inds = np.unique(topo)
    return coords[inds], remap_topo(topo, inds_to_invmap_as_dict(inds))


@njit(nogil=True, parallel=True, cache=__cache)
def explode_mesh_bulk(coords: ndarray, topo: ndarray):
    nE, nNE = topo.shape
    nD = coords.shape[1]
    coords_ = np.zeros((nE*nNE, nD), dtype=coords.dtype)
    topo_ = np.zeros_like(topo)
    for i in prange(nE):
        ii = i*nNE
        for j in prange(nNE):
            coords_[ii + j] = coords[topo[i, j]]
            topo_[i, j] = ii + j
    return coords_, topo_


@njit(nogil=True, parallel=True, cache=__cache)
def explode_mesh_data_bulk(coords: ndarray, topo: ndarray, data: ndarray):
    nE, nNE = topo.shape
    nD = coords.shape[1]
    coords_ = np.zeros((nE*nNE, nD), dtype=coords.dtype)
    topo_ = np.zeros_like(topo)
    data_ = np.zeros(nE*nNE, dtype=coords.dtype)
    for i in prange(nE):
        ii = i*nNE
        for j in prange(nNE):
            coords_[ii + j] = coords[topo[i, j]]
            data_[ii + j] = data[i, topo[i, j]]
            topo_[i, j] = ii + j
    return coords_, topo_, data_


@njit(nogil=True, parallel=True, cache=__cache)
def avg_cell_data1d_bulk(data: np.ndarray, topo: np.ndarray):
    nE, nNE = topo.shape
    nD = data.shape[1]
    res = np.zeros((nE, nD), dtype=data.dtype)
    for iE in prange(nE):
        for jNE in prange(nNE):
            ind = topo[iE, jNE]
            for kD in prange(nD):
                res[iE, kD] += data[ind, kD]
        res[iE, :] /= nNE
    return res


def avg_cell_data(data: np.ndarray, topo: np.ndarray, squeeze=True):
    nR = len(data.shape)
    if nR == 2:
        res = avg_cell_data1d_bulk(matrixform(data), topo)
    if squeeze:
        return np.squeeze(res)
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def jacobian_matrix_bulk(dshp: ndarray, ecoords: ndarray):
    nE = ecoords.shape[0]
    nG, _, nD = dshp.shape
    jac = np.zeros((nE, nG, nD, nD), dtype=dshp.dtype)
    for iE in prange(nE):
        points = ecoords[iE].T
        for iG in prange(nG):
            jac[iE, iG] = points @ dshp[iG]
    return jac


@njit(nogil=True, parallel=True, cache=__cache)
def center_of_points(coords : ndarray):
    res = np.zeros(coords.shape[1], dtype=coords.dtype)
    for i in prange(res.shape[0]):
        res[i] = np.mean(coords[:, i])
    return res


@njit(nogil=True, cache=__cache)
def centralize(coords : ndarray):
    nD = coords.shape[1]
    center = center_of_points(coords)
    coords[:, 0] -= center[0]
    coords[:, 1] -= center[1]
    if nD > 2:
        coords[:, 2] -= center[2]
    return coords


@njit(nogil=True, parallel=True, cache=__cache)
def lengths_of_lines(coords: ndarray, topo: ndarray):
    nE, nNE = topo.shape
    res = np.zeros(nE, dtype=coords.dtype)
    for i in prange(nE):
        res[i] = norm(coords[topo[i, nNE-1]] - coords[topo[i, 0]])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def lengths_of_lines2(ecoords: ndarray):
    nE, nNE = ecoords.shape[:2]
    res = np.zeros(nE, dtype=ecoords.dtype)
    for i in prange(nE):
        res[i] = norm(ecoords[i, nNE-1] - ecoords[i, 0])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def distances_of_points(coords: ndarray):
    nP = coords.shape[0]
    res = np.zeros(nP, dtype=coords.dtype)
    for i in prange(1, nP):
        res[i] = norm(coords[i] - coords[i-1])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def pcoords_to_coords(pcoords: ndarray, ecoords: ndarray):
    nP = pcoords.shape[0]
    nE = ecoords.shape[0]
    nX = nE * nP
    res = np.zeros((nX, ecoords.shape[2]), dtype=ecoords.dtype)
    for iE in prange(nE):
        for jP in prange(nP):
            res[iE * nP + jP] = ecoords[iE, 0] * (1-pcoords[jP]) \
                + ecoords[iE, -1] * pcoords[jP]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def norms(a: np.ndarray):
    nI = len(a)
    res = np.zeros(nI, dtype=a.dtype)
    for iI in prange(len(a)):
        res[iI] = np.dot(a[iI], a[iI])
    return np.sqrt(res)





