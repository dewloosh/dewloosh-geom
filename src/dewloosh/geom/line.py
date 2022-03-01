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
           
    def volume(self, *args, **kwargs):
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs):
        lengths = self.lengths(*args, **kwargs)
        if 'area' in self.fields:
            areas = self.area.to_numpy()
            return lengths * areas
        else:
            return lengths