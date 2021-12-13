# -*- coding: utf-8 -*-
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
    vtkCellType = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HexaHedron(PolyHedron):

    NNODE = 8
    vtkCellType = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TriquadraticHexaHedron(PolyHedron):

    NNODE = 27
    vtkCellType = 29

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Wedge(PolyHedron):

    NNODE = 6
    vtkCellType = 13

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiquadraticWedge(PolyHedron):

    NNODE = 18
    vtkCellType = 32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
