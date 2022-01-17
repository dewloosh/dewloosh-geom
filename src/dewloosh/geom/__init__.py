# -*- coding: utf-8 -*-
__version__ = "0.0.dev8"
__description__ = "A pure python package to build, manipulate and analyze polygonal meshes."
import dewloosh.geom.topo.tr as tr
import dewloosh.geom.topo.topodata as topodata
from dewloosh.geom.rgrid import *
from dewloosh.geom.mesh1d import mesh1d_uniform
from dewloosh.geom.polydata import PolyData as PolyData
from dewloosh.geom.T3 import T3 as Tri
from dewloosh.geom.Q4 import Q4 as Quad
from dewloosh.geom.Q9 import Q9
from dewloosh.geom.T6 import T6
from dewloosh.geom.H8 import H8
from dewloosh.geom.H27 import H27
from dewloosh.geom.TET4 import TET4 as Tetra
from dewloosh.geom.tri.trimesh import TriMesh
from dewloosh.geom.tet.tetmesh import TetMesh
