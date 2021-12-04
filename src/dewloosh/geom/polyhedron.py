# -*- coding: utf-8 -*-
import vtk
from dewloosh.geom.cell import PolyCell3d
import numpy as np


class PolyHedron(PolyCell3d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def to_tetrahedra(cls, coords, topo):
        raise NotImplementedError

    def volume(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.pointdata.x.to_numpy()
        if topo is None:
            topo = self.nodes.to_numpy()
        return np.sum(self.volumes(coords, topo))


class TetraHedron(PolyHedron):

    NNODE = 4
    vtkCellType = vtk.VTK_TETRA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HexaHedron(PolyHedron):

    NNODE = 8
    vtkCellType = vtk.VTK_HEXAHEDRON

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TriquadraticHexaHedron(PolyHedron):

    NNODE = 27
    vtkCellType = vtk.VTK_TRIQUADRATIC_HEXAHEDRON

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Wedge(PolyHedron):

    NNODE = 6
    vtkCellType = vtk.VTK_WEDGE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiquadraticWedge(PolyHedron):

    NNODE = 18
    vtkCellType = vtk.VTK_BIQUADRATIC_QUADRATIC_WEDGE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
