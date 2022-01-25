# -*- coding: utf-8 -*-
from dewloosh.geom import PolyData
from dewloosh.geom import H8
from dewloosh.geom.rgrid import grid
from dewloosh.geom.utils import detach_mesh_bulk
import numpy as np
from numpy import ndarray


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


if __name__ == '__main__':
    d = 100.0
    h = 0.8
    a = 1.5
    b = 0.5
    
    coords, topo = voxelize_cylinder(radius=[b/2, a/2], height=h, size=h/20)
    PolyData(coords=coords, topo=topo, celltype=H8).plot()