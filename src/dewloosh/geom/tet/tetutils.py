# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def tet_vol_bulk(ecoords: np.ndarray):
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    for i in prange(nE):
        v1 = ecoords[i, 1] - ecoords[i, 0]
        v2 = ecoords[i, 2] - ecoords[i, 0]
        v3 = ecoords[i, 3] - ecoords[i, 0]
        res[i] = np.dot(np.cross(v1, v2), v3)
    return res / 6


@njit(nogil=True, parallel=True, cache=__cache)
def extrude_T3_TET4(points: ndarray, triangles: ndarray, 
                    h: float=1.0, zres: int=1):
    nT = triangles.shape[0]
    nP = points.shape[0]
    nE = nT * zres * 3
    nC = nP * (zres + 1)
    coords = np.zeros((nC, 3), dtype=points.dtype)
    topo = np.zeros((nE, 4), dtype=triangles.dtype)
    coords[:nP, :2] = points[:, :2]        
    for i in prange(zres):
        coords[nP * (i + 1) : nP * (i + 2), :2] = points[:, :2] 
        coords[nP * (i + 1) : nP * (i + 2), 2] = h * (i + 1) / zres
        for j in prange(nT):
            id = i * nT * 3 + j * 3
            i_0, j_0, k_0 = triangles[j] + i * nP
            i_1, j_1, k_1 = triangles[j] + (i + 1) * nP
            #
            topo[id, 0] = i_0
            topo[id, 1] = j_0
            topo[id, 2] = k_0
            topo[id, 3] = k_1
            #
            topo[id + 1, 0] = i_0
            topo[id + 1, 1] = j_0
            topo[id + 1, 2] = k_1
            topo[id + 1, 3] = j_1
            #
            topo[id + 2, 0] = i_0
            topo[id + 2, 1] = j_1
            topo[id + 2, 2] = k_1
            topo[id + 2, 3] = i_1
    return coords, topo


if __name__ == '__main__':
    from dewloosh.geom.tri.trimesh import circular_disk
    from dewloosh.geom import PolyData
    from dewloosh.geom.TET4 import TET4
    from dewloosh.geom.utils import detach_mesh_bulk
        
    n_angles = 120
    n_radii = 60
    min_radius = 5
    max_radius = 25
    h = 20
    zres = 20

    points, triangles = \
        circular_disk(n_angles, n_radii, min_radius, max_radius)
    
    points, triangles = detach_mesh_bulk(points, triangles)
        
    coords, topo = extrude_T3_TET4(points, triangles, h, zres)
    
    tetmesh = PolyData(coords=coords, topo=topo, celltype=TET4)
    tetmesh.plot()