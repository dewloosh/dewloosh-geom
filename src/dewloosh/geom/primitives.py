# -*- coding: utf-8 -*-
from dewloosh.geom import PolyData
from dewloosh.solid.fem import H8, TET4
from dewloosh.geom.space import CartesianFrame
import numpy as np
from numpy import ndarray


def Cylinder(shape, size=None, *args, regular=True, voxelize=False, 
             celltype=None, **kwargs):
    etype = None
    if isinstance(size, float):
        voxelize=True
    if voxelize:
        regular = True
        etype = 'H8'
    frame = CartesianFrame(dim=3)
    radius, angle, h = shape
    if isinstance(radius, int):
        radius = np.array([0, radius])
    elif not isinstance(radius, ndarray):
        radius = np.array(radius)
    if celltype is None:
        celltype = H8 if voxelize else TET4
    etype = celltype.__label__ if etype is None else etype
    if regular:
        if etype == 'H8':
            if voxelize:
                from dewloosh.geom.voxelize import voxelize_cylinder
                coords, topo = \
                    voxelize_cylinder(radius=radius, height=h, size=size)
        elif etype == 'TET4':
            from dewloosh.geom.tri.triang import circular_disk
            from dewloosh.geom.utils import detach_mesh_bulk
            from dewloosh.geom.extrude import extrude_T3_TET4
            min_radius, max_radius = radius
            n_radii, n_angles, n_z = size
            points, triangles = \
                circular_disk(n_angles, n_radii, min_radius, max_radius)
            points, triangles = detach_mesh_bulk(points, triangles)
            coords, topo = extrude_T3_TET4(points, triangles, h, n_z)
    else:
        import tetgen
        import pyvista as pv
        (a, b), angle, h = shape
        n_radii, n_angles, n_z = size
        cyl = pv.CylinderStructured(center=(0.0, 0.0, h/2), direction=(0.0, 0.0, 1.0),
                                    radius=np.linspace(b/2, a/2, n_radii), height=h, 
                                    theta_resolution=n_angles, z_resolution=n_z)
        cyl_surf = cyl.extract_surface().triangulate()
        tet = tetgen.TetGen(cyl_surf)
        tet.tetrahedralize(order=1, mindihedral=10, minratio=1.1, quality=True)
        grid = tet.grid
        coords = np.array(grid.points).astype(float)
        topo = grid.cells_dict[10].astype(np.int32)
    return PolyData(coords=coords, topo=topo, celltype=celltype, frame=frame)