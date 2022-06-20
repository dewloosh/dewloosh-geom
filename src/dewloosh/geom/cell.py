# -*- coding: utf-8 -*-
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy as np
from numpy import ndarray

from dewloosh.math.array import atleast1d
from dewloosh.math.utils import to_range

from .celldata import CellData
from .utils import jacobian_matrix_bulk, points_of_cells, pcoords_to_coords_1d


class PolyCell(CellData):

    NNODE = None
    NDIM = None

    def __init__(self, *args, topo: ndarray=None, i: ndarray=None, **kwargs):
        if isinstance(topo, ndarray):
            kwargs['nodes'] = topo
        if isinstance(i, ndarray):
            kwargs['id'] = i
        super().__init__(*args, **kwargs)
        
    def jacobian_matrix(self, *args, dshp=None, ecoords=None, topo=None, **kwargs):
        ecoords = self.local_coordinates(topo=topo) if ecoords is None else ecoords
        return jacobian_matrix_bulk(dshp, ecoords)
    
    def jacobian(self, *args, jac=None, **kwargs):
        return np.linalg.det(jac)
    
    def points_of_cells(self, *args, target=None, **kwargs):
        assert target is None
        topo = kwargs.get('topo', self.nodes.to_numpy())
        coords = kwargs.get('coords', self.pointdata.x.to_numpy())
        return points_of_cells(coords, topo)
                    
    def local_coordinates(self, *args, **kwargs):
        frames = kwargs.get('frames', self.frames.to_numpy())
        topo = kwargs.get('_topo', self.nodes.to_numpy())
        coords = self.pointdata.x.to_numpy()
        return points_of_cells(coords, topo, local_axes=frames)
    
    def coords(self, *args, **kwargs):
        return self.points_of_cells(*args, **kwargs)
    
    
class PolyCell1d(PolyCell):

    NDIM = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # NOTE The functionality of `pcoords_to_coords_1d` needs to be generalized 
    # for higher order cells.    
    def points_of_cells(self, *args, points=None, cells=None, target='global', 
                        rng=None, flatten=False, **kwargs):
        if isinstance(target, str):
            assert target.lower() in ['global', 'g']
        else:
            raise NotImplementedError
        topo = kwargs.get('topo', self.nodes.to_numpy())
        coords = kwargs.get('coords', self.pointdata.x.to_numpy())
        ecoords = points_of_cells(coords, topo)
        if points is None and cells is None:
            return ecoords
        
        # points or cells is not None
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id.to_numpy())
            cells = atleast1d(cells[conds])
            if len(cells) == 0:
                return {} 
            ecoords = ecoords[cells]
            topo = topo[cells]
        else:
            cells = np.s_[:]
            
        if points is None:
            points = np.array(self.lcoords()).flatten()
            rng = [-1, 1]
        else:
            rng = np.array([0, 1]) if rng is None else np.array(rng)
            
        points, rng = to_range(points, source=rng, target=[0, 1]).flatten(), [0, 1]
        datacoords = pcoords_to_coords_1d(points, ecoords)  # (nE * nP, nD)
        
        if not flatten:
            nE = ecoords.shape[0]
            nP = points.shape[0]
            datacoords = datacoords.reshape(nE, nP, datacoords.shape[-1])  # (nE, nP, nD)
        
        # values : (nE, nP, nDOF, nRHS) or (nE, nP * nDOF, nRHS)
        if isinstance(cells, slice):
            # results are requested on all elements 
            data = datacoords
        elif isinstance(cells, Iterable):
            data = {c : datacoords[i] for i, c in enumerate(cells)}                    
        else:
            raise TypeError("Invalid data type <> for cells.".format(type(cells)))    
        
        return data
                
    
class PolyCell2d(PolyCell):

    NDIM = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
class PolyCell3d(PolyCell):

    NDIM = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        