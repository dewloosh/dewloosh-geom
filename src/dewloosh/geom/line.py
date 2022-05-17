# -*- coding: utf-8 -*-
from .utils import lengths_of_lines
from .cell import PolyCell1d
import numpy as np


__all__ = ['Line']


class Line(PolyCell1d):
    
    NNODE = 2
    vtkCellType = 3
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                        
    def lengths(self, *args, coords=None, topo=None, **kwargs):
        coords = self.pointdata.x.to_numpy() if coords is None else coords
        topo = self.nodes.to_numpy() if topo is None else topo
        return lengths_of_lines(coords, topo)
    
    def length(self, *args, **kwargs):
        return np.sum(self.length(*args, **kwargs))
    
    def areas(self, *args, **kwargs):
        if 'areas' in self.fields:
            return self['areas'].to_numpy()
        else:
            return np.ones((len(self)))
    
    def area(self, *args, **kwargs):
        return np.sum(self.areas(*args, **kwargs))
    
    def volume(self, *args, **kwargs):
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs):
        return self.lengths(*args, **kwargs) * self.areas(*args, **kwargs)
        
        
class QuadraticLine(Line):
    
    NNODE = 3
    vtkCellType = None
    