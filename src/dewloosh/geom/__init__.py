# -*- coding: utf-8 -*-
__version__ = "0.0.1-alpha1"
__description__ = "A Python package to build, manipulate and analyze polygonal meshes."

from dewloosh.geom.space import PointCloud
from dewloosh.geom.space import CartesianFrame
from dewloosh.geom.tri.triang import triangulate
from dewloosh.geom.rgrid import grid, Grid
from dewloosh.geom.tri.trimesh import TriMesh
from dewloosh.geom.tet.tetmesh import TetMesh
from dewloosh.geom.polydata import PolyData