# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from .polydata import PolyData
from .cells import H8
from .rgrid import grid
from .topo import detach_mesh_bulk


def voxelize_cylinder(radius: ndarray, height: float, size: float):
    if isinstance(radius, int):
        radius = np.array([0, radius])
    elif not isinstance(radius, ndarray):
        radius = np.array(radius)
    nXY = int(np.ceil(2 * radius[1] / size))
    nZ = int(np.ceil(height / size))
    Lxy, Lz = 2 * radius[1], height
    coords, topo = grid(size=(Lxy, Lxy, Lz), shape=(nXY, nXY, nZ), 
                        eshape='H8', centralize=True)
    c = PolyData(coords=coords, topo=topo, celltype=H8).centers()
    r = (c[:, 0]**2 + c[:, 1]**2)**(1/2)
    cond = (r <= radius[1]) & (r >= radius[0])
    inds = np.where(cond)[0]
    return detach_mesh_bulk(coords, topo[inds])