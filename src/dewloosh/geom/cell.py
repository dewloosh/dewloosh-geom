# -*- coding: utf-8 -*-
from dewloosh.geom.celldata import CellData
from dewloosh.geom.utils import jacobian_matrix_bulk
import numpy as np


class PolyCell(CellData):

    NNODE = None
    NDIM = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def jacobian_matrix(self, *args, dshp=None, ecoords=None, topo=None, **kwargs):
        ecoords = self.local_coordinates(topo=topo) if ecoords is None else ecoords
        return jacobian_matrix_bulk(dshp, ecoords)
    
    def jacobian(self, *args, jac=None, **kwargs):
        return np.linalg.det(jac)


class PolyCell1d(PolyCell):

    NDIM = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
class PolyCell2d(PolyCell):

    NDIM = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
class PolyCell3d(PolyCell):

    NDIM = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        