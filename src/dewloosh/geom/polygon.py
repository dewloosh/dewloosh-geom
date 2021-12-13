# -*- coding: utf-8 -*-
from dewloosh.geom.utils import cell_coords_bulk
from dewloosh.geom.tri.triutils import area_tri_bulk
from dewloosh.geom.topo.tr import T6_to_T3, Q4_to_T3, Q9_to_Q4
from dewloosh.geom.cell import PolyCell2d
import numpy as np


class PolyGon(PolyCell2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_triangles(self, coords, topo):
        raise NotImplementedError

    def area(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.pointdata.x.to_numpy()
        if topo is None:
            topo = self.nodes.to_numpy()
        return np.sum(self.areas(coords, topo))

    def areas(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.pointdata.x.to_numpy()
        if topo is None:
            topo = self.nodes.to_numpy()
        areas = area_tri_bulk(cell_coords_bulk(
            *self.to_triangles(coords, topo)))
        res = np.sum(areas.reshape(topo.shape[0], int(
            len(areas)/topo.shape[0])), axis=1)
        return np.squeeze(res)

    def volume(self, *args, **kwargs):
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs):
        areas = self.areas(*args, **kwargs)
        if 't' in self.fields:
            t = self.t.to_numpy()
            return areas * t
        else:
            return areas


class Triangle(PolyGon):

    NNODE = 3
    vtkCellType = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def areas(self, *args, coords=None, topo=None, **kwargs):
        if coords is None:
            coords = self.pointdata.x.to_numpy()
        if topo is None:
            topo = self.nodes.to_numpy()
        return area_tri_bulk(cell_coords_bulk(coords, topo))


class QuadraticTriangle(PolyGon):

    NNODE = 6
    vtkCellType = 22

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_triangles(self, coords=None, topo=None, data=None):
        if coords is None:
            coords = self.pointdata.x.to_numpy()
        if topo is None:
            topo = self.nodes.to_numpy()
        return T6_to_T3(coords, topo, data)


class Quadrilateral(PolyGon):

    NNODE = 4
    vtkCellType = 9

    def to_triangles(self, coords=None, topo=None, data=None):
        if coords is None:
            coords = self.pointdata.x.to_numpy()
        if topo is None:
            topo = self.nodes.to_numpy()
        return Q4_to_T3(coords, topo, data)


class BiQuadraticQuadrilateral(PolyGon):

    NNODE = 9
    vtkCellType = 28

    def to_triangles(self, coords=None, topo=None, data=None):
        if coords is None:
            coords = self.pointdata.x.to_numpy()
        if topo is None:
            topo = self.nodes.to_numpy()
        return Q4_to_T3(*Q9_to_Q4(coords, topo, data))


if __name__ == '__main__':
    pass
