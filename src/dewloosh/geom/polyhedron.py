# -*- coding: utf-8 -*-
from dewloosh.geom.cell import PolyCell3d
from dewloosh.geom.tet.tetutils import tet_vol_bulk
from dewloosh.geom.utils import cell_coords_bulk
import numpy as np


class PolyHedron(PolyCell3d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tetrahedra(self, coords, topo):
        raise NotImplementedError

    def volume(self, *args, coords=None, topo=None, **kwargs):
        coords = self.pointdata.x.to_numpy() if coords is None else coords
        topo = self.nodes.to_numpy() if topo is None else topo
        return np.sum(self.volumes(coords, topo))
    
    def volumes(self, *args, coords=None, topo=None, **kwargs):
        coords = self.pointdata.x.to_numpy() if coords is None else coords
        topo = self.nodes.to_numpy() if topo is None else topo
        volumes = tet_vol_bulk(cell_coords_bulk(
            *self.to_tetrahedra(coords, topo)))
        res = np.sum(volumes.reshape(topo.shape[0], int(
            len(volumes)/topo.shape[0])), axis=1)
        return np.squeeze(res)


class TetraHedron(PolyHedron):

    NNODE = 4
    vtkCellType = 10
    __label__ = 'TET4'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def to_tetrahedra(self, coords, topo):
        return coords, topo


class HexaHedron(PolyHedron):

    NNODE = 8
    vtkCellType = 12
    __label__ = 'H8'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TriquadraticHexaHedron(PolyHedron):

    NNODE = 27
    vtkCellType = 29
    __label__ = 'H27'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Wedge(PolyHedron):

    NNODE = 6
    vtkCellType = 13
    __label__ = 'W6'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BiquadraticWedge(PolyHedron):

    NNODE = 18
    vtkCellType = 32
    __label__ = 'W18'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
