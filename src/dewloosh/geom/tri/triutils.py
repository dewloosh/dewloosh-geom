# -*- coding: utf-8 -*-
from dewloosh.geom.utils import cell_coords_bulk, cell_coords
from dewloosh.math.linalg import normalize
import numpy as np
from numba import njit, prange, vectorize
__cache = True


@njit(nogil=True, cache=__cache)
def monoms_tri_loc(lcoord: np.ndarray):
    return np.array([1, lcoord[0], lcoord[1]], dtype=lcoord.dtype)


@njit(nogil=True, cache=__cache)
def monoms_tri_loc_bulk(lcoord: np.ndarray):
    res = np.ones((lcoord.shape[0], 3), dtype=lcoord.dtype)
    res[:, 1] = lcoord[:, 0]
    res[:, 2] = lcoord[:, 1]
    return res


@njit(nogil=True, cache=__cache)
def lcoords_tri():
    return np.array([[0., 0.], [1., 0.], [0., 1.]])


@njit(nogil=True, cache=__cache)
def lcenter_tri():
    return np.array([1/3, 1/3])


@njit(nogil=True, cache=__cache)
def ncenter_tri():
    return np.array([1/3, 1/3, 1/3])


@njit(nogil=True, cache=__cache)
def shp_tri_loc(lcoord: np.ndarray):
    return np.array([1 - lcoord[0] - lcoord[1], lcoord[0], lcoord[1]])


@njit(nogil=True, parallel=True, cache=__cache)
def shape_function_matrix_tri_loc(lcoord: np.ndarray, nDOFN=2, nNODE=3):
    eye = np.eye(nDOFN, dtype=lcoord.dtype)
    shp = shp_tri_loc(lcoord)
    res = np.zeros((nDOFN, nNODE * nDOFN), dtype=lcoord.dtype)
    for i in prange(nNODE):
        res[:, i * nNODE: (i+1) * nNODE] = eye * shp[i]
    return res


@njit(nogil=True, cache=__cache)
def center_tri_2d(ecoords: np.ndarray):
    return np.array([np.mean(ecoords[:, 0]), np.mean(ecoords[:, 1])],
                    dtype=ecoords.dtype)


@njit(nogil=True, cache=__cache)
def center_tri_3d(ecoords: np.ndarray):
    return np.array([np.mean(ecoords[:, 0]), np.mean(ecoords[:, 1]),
                     np.mean(ecoords[:, 2])], dtype=ecoords.dtype)


@njit(nogil=True, cache=__cache)
def area_tri(ecoords: np.ndarray):
    """
    Returnsthe the signed area of a 3-noded triangle.
    """
    A = (ecoords[1, 0]*ecoords[2, 1] - ecoords[2, 0]*ecoords[1, 1]) + \
        (ecoords[2, 0]*ecoords[0, 1] - ecoords[0, 0]*ecoords[2, 1]) + \
        (ecoords[0, 0]*ecoords[1, 1] - ecoords[1, 0]*ecoords[0, 1])
    return A/2


@njit(nogil=True, parallel=True, cache=__cache)
def areas_tri(ecoords: np.ndarray):
    A = 0.
    nE = len(ecoords)
    for i in prange(nE):
        A += (ecoords[i, 1, 0]*ecoords[i, 2, 1] -
              ecoords[i, 2, 0]*ecoords[i, 1, 1]) + \
             (ecoords[i, 2, 0]*ecoords[i, 0, 1] -
              ecoords[i, 0, 0]*ecoords[i, 2, 1]) + \
             (ecoords[i, 0, 0]*ecoords[i, 1, 1] -
              ecoords[i, 1, 0]*ecoords[i, 0, 1])
    return A/2


@njit(nogil=True, parallel=True, cache=__cache)
def area_tri_bulk(ecoords: np.ndarray):
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    for i in prange(nE):
        res[i] = (ecoords[i, 1, 0]*ecoords[i, 2, 1] -
                  ecoords[i, 2, 0]*ecoords[i, 1, 1]) + \
                 (ecoords[i, 2, 0]*ecoords[i, 0, 1] -
                  ecoords[i, 0, 0]*ecoords[i, 2, 1]) + \
                 (ecoords[i, 0, 0]*ecoords[i, 1, 1] -
                  ecoords[i, 1, 0]*ecoords[i, 0, 1])
    return res/2


@vectorize("f8(f8, f8, f8, f8, f8, f8)", target='parallel', cache=__cache)
def areas_tri_u(x1, y1, x2, y2, x3, y3: np.ndarray):
    return (x2*y3 - x3*y2 + x3*y1 - x1*y3 + x1*y2 - x2*y1)/2


@vectorize("f8(f8, f8, f8, f8, f8, f8)", target='parallel', cache=__cache)
def areas_tri_u2(x1, x2, x3, y1, y2, y3: np.ndarray):
    return (x2*y3 - x3*y2 + x3*y1 - x1*y3 + x1*y2 - x2*y1)/2


@njit(nogil=True, cache=__cache)
def loc_to_glob_tri(lcoord: np.ndarray, gcoords: np.ndarray,
                    dtype=np.float32):
    return gcoords.T @ shp_tri_loc(lcoord, dtype)


@njit(nogil=True, cache=__cache)
def glob_to_loc_tri(gcoord: np.ndarray, gcoords: np.ndarray):
    monoms = monoms_tri_loc_bulk(gcoords)
    coeffs = np.linalg.inv(monoms)
    shp = coeffs @ monoms_tri_loc(gcoord)
    return lcoords_tri().T @ shp


@njit(nogil=True, cache=__cache)
def glob_to_nat_tri(gcoord: np.ndarray, ecoords: np.ndarray):
    x, y = gcoord[0:2]
    (x1, x2, x3), (y1, y2, y3) = ecoords[:, 0], ecoords[:, 1]
    A2 = np.abs(x2*(y3-y1) + x1*(y2-y3) + x3*(y1-y2))
    n1 = (x2*y3 - x*y3 - x3*y2 + x*y2 + x3*y - x2*y)/A2
    n2 = (x*y3 - x1*y3 + x3*y1 - x*y1 - x3*y + x1*y)/A2
    return np.array([n1, n2, 1 - n1 - n2], dtype=gcoord.dtype)


@njit(nogil=True, cache=__cache)
def nat_to_glob_tri(ncoord: np.ndarray, ecoords: np.ndarray):
    return ecoords.T @ ncoord


@njit(nogil=True, cache=__cache)
def loc_to_nat_tri(lcoord: np.ndarray):
    return shp_tri_loc(lcoord)


@njit(nogil=True, cache=__cache)
def nat_to_loc_tri(acoord: np.ndarray):
    return lcoords_tri.T @ acoord


def offset_tri(coords: np.ndarray, topo: np.ndarray, data: np.ndarray,
               *args, **kwargs):
    if isinstance(data, np.ndarray):
        alpha = np.abs(data)
        amax = alpha.max()
        if amax > 1.0:
            alpha /= amax
        return _offset_tri_(coords, topo, alpha)
    elif isinstance(data, float):
        alpha = min(abs(data), 1.0)
        return offset_tri_uniform(coords, topo, alpha)
    else:
        raise RuntimeError


@njit(nogil=True, cache=__cache)
def offset_tri_uniform(coords: np.ndarray, topo: np.ndarray, alpha=0.9):
    cellcoords = cell_coords_bulk(coords, topo)
    ncenter = ncenter_tri(coords.dtype)
    eye = np.eye(3, dtype=coords.dtype)
    ncoords = ncenter + (eye - ncenter) * alpha
    nE = len(topo)
    res = np.zeros(cellcoords.shape, dtype=cellcoords.dtype)
    ncoords = ncoords.astype(cellcoords.dtype)
    for iE in prange(nE):
        res[iE, :, :] = ncoords @ cellcoords[iE]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _offset_tri_(coords: np.ndarray, topo: np.ndarray, alpha: np.ndarray):
    cellcoords = cell_coords_bulk(coords, topo)
    ncenter = ncenter_tri()
    dn = (np.eye(3, dtype=coords.dtype) - ncenter)
    nE = len(topo)
    res = np.zeros(cellcoords.shape, dtype=cellcoords.dtype)
    alpha = alpha.astype(cellcoords.dtype)
    for iE in prange(nE):
        ncoords = ncenter + dn * alpha[iE]
        res[iE, :, :] = ncoords @ cellcoords[iE]
    return res


def edges_tri(triangles: np.ndarray):
    shp = triangles.shape
    if len(shp) == 2:
        return _edges_tri(triangles)
    elif len(shp) == 3:
        return _edges_tri_pop(triangles)
    else:
        raise NotImplementedError


@njit(nogil=True, cache=__cache)
def _edges_tri(triangles: np.ndarray):
    nE = len(triangles)
    edges = np.zeros((nE, 3, 2), dtype=triangles.dtype)
    edges[:, 0, 0] = triangles[:, 0]
    edges[:, 0, 1] = triangles[:, 1]
    edges[:, 1, 0] = triangles[:, 1]
    edges[:, 1, 1] = triangles[:, 2]
    edges[:, 2, 0] = triangles[:, 2]
    edges[:, 2, 1] = triangles[:, 0]
    return edges


@njit(nogil=True, parallel=True, cache=__cache)
def _edges_tri_pop(triangles: np.ndarray):
    nPop, nE, _ = triangles.shape
    res = np.zeros((nPop, nE, 3, 2), dtype=triangles.dtype)
    for i in prange(nPop):
        res[i] = _edges_tri(triangles[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tri_glob_to_loc(points: np.ndarray, triangles: np.ndarray):
    nE = triangles.shape[0]
    tr = np.zeros((nE, 3, 3), dtype=points.dtype)
    res = np.zeros((nE, 3, 2), dtype=points.dtype)
    centers = np.zeros((nE, 3), dtype=points.dtype)
    for iE in prange(nE):
        centers[iE] = center_tri_3d(cell_coords(points, triangles[iE]))
        tr[iE, 0, :] = normalize(
            points[triangles[iE, 1]] - points[triangles[iE, 0]])
        tr[iE, 1, :] = normalize(
            points[triangles[iE, 2]] - points[triangles[iE, 0]])
        tr[iE, 2, :] = np.cross(tr[iE, 0, :], tr[iE, 1, :])
        tr[iE, 1, :] = np.cross(tr[iE, 2, :], tr[iE, 0, :])
        for jN in prange(3):
            vj = points[triangles[iE, jN]] - centers[iE]
            res[iE, jN, 0] = np.dot(tr[iE, 0, :], vj)
            res[iE, jN, 1] = np.dot(tr[iE, 1, :], vj)
    return res, centers, tr


if __name__ == '__main__':
    from dewloosh.geom.trimesh import triangulation

    points, triangles, triobj = triangulation(size=(800, 600),
                                              shape=(10, 10))
    tricoords = cell_coords_bulk(points, triangles)

    area0 = 800 * 600

    area1 = np.sum(area_tri_bulk(tricoords))

    area2 = areas_tri(tricoords)

    x1 = tricoords[:, 0, 0]
    x2 = tricoords[:, 1, 0]
    x3 = tricoords[:, 2, 0]
    y1 = tricoords[:, 0, 1]
    y2 = tricoords[:, 1, 1]
    y3 = tricoords[:, 2, 1]
    area3 = np.sum(areas_tri_u2(x1, x2, x3, y1, y2, y3))

    tri_glob_to_loc(points, triangles)
