# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray

from .polydata import PolyData
from .tri.trimesh import TriMesh
from .cells import H8, TET4, T3
from .space import CartesianFrame
from .tri.triang import triangulate
from .topo import detach_mesh_bulk
from .extrude import extrude_T3_TET4
from .voxelize import voxelize_cylinder


def circular_disk(nangles, nradii, rmin, rmax):
    """
    Returns the triangulation of a circular disk.
    
    Parameters
    ----------
    nangles : int
        Number of subdivisions in radial direction.
        
    nradii : int
        Number of subdivisions in circumferential direction.
        
    rmin : float
        Inner radius. Can be zero.
    
    rmax : float
        Outer radius.
    """
    radii = np.linspace(rmin, rmax, nradii)
    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], nradii, axis=1)
    angles[:, 1::2] += np.pi / nangles
    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    nP = len(x)
    points = np.stack([x, y], axis=1)
    *_, triang = triangulate(points=points, backend='mpl')
    #triang = tri.Triangulation(x, y)
    # Mask off unwanted triangles.
    triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                            y[triang.triangles].mean(axis=1))
                    < rmin)
    triangles = triang.get_masked_triangles()
    points = np.stack((triang.x, triang.y, np.zeros(nP)), axis = 1)
    points, triangles = detach_mesh_bulk(points, triangles)
    return points, triangles


def CircularDisk(*args, **kwargs):
    points, triangles = circular_disk(*args, **kwargs)
    return TriMesh(points=points, triangles=triangles, celltype=T3)


def cylinder(shape, size=None, *args, regular=True, voxelize=False, 
             celltype=None, **kwargs):
    etype = None
    if isinstance(size, float):
        voxelize=True
    if voxelize:
        regular = True
        etype = 'H8'
    radius, angle, h = shape
    if isinstance(radius, int):
        radius = np.array([0, radius])
    elif not isinstance(radius, ndarray):
        radius = np.array(radius)
    etype = celltype.__label__ if etype is None else etype
    if voxelize:
        size_ = (radius[1] - radius[0]) / size[0]
        coords, topo = voxelize_cylinder(radius=radius, height=h, size=size_)
    else:
        if regular:
            if etype == 'TET4':
                min_radius, max_radius = radius
                n_radii, n_angles, n_z = size
                points, triangles = circular_disk(n_angles, n_radii, min_radius, max_radius)
                coords, topo = extrude_T3_TET4(points, triangles, h, n_z)
            else:
                raise NotImplementedError("Celltype not supported!")
        else:
            import tetgen
            import pyvista as pv
            (rmin, rmax), angle, h = shape
            n_radii, n_angles, n_z = size
            cyl = pv.CylinderStructured(center=(0.0, 0.0, h/2), direction=(0.0, 0.0, 1.0),
                                        radius=np.linspace(rmin, rmax, n_radii), height=h, 
                                        theta_resolution=n_angles, z_resolution=n_z)
            cyl_surf = cyl.extract_surface().triangulate()
            tet = tetgen.TetGen(cyl_surf)
            tet.tetrahedralize(order=1, mindihedral=10, minratio=1.1, quality=True)
            grid = tet.grid
            coords = np.array(grid.points).astype(float)
            topo = grid.cells_dict[10].astype(np.int32)
    return coords, topo


def Cylinder(*args, celltype=None, voxelize=False, **kwargs):
    if celltype is None:
        celltype = H8 if voxelize else TET4
    coords, topo = cylinder(*args, celltype=celltype, voxelize=voxelize, **kwargs)
    frame = CartesianFrame(dim=3)
    return PolyData(coords=coords, topo=topo, celltype=celltype, frame=frame)


if __name__ == '__main__':    
    trimesh = TriMesh(size=(800, 600), shape=(10, 10))
    trimesh.plot()
    
    n_angles = 120
    n_radii = 60
    min_radius = 5
    max_radius = 25

    disk = circular_disk(n_angles, n_radii, min_radius, max_radius)
    disk.plot()