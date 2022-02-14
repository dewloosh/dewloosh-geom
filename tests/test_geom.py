# -*- coding: utf-8 -*-
import numpy as np
from hypothesis import given, strategies as st
import unittest

from dewloosh.math.linalg import Vector

from dewloosh.geom import TriMesh, PolyData, grid, PointCloud, CartesianFrame


def test_coord_tr_1(i, a):
    A = CartesianFrame(dim=3)
    coords = PointCloud([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ], frame=A)
    amounts = [0, 0, 0]
    amounts[i] = a * np.pi / 180
    B = A.orient_new('Body', amounts, 'XYZ')
    arr_new = Vector(coords.array, frame=A).view(B)
    coords_new = Vector(arr_new, frame=B)
    return np.max(np.abs(coords_new.view() - coords.view())) < 1e-8


class TestCoords(unittest.TestCase):

    @given(st.integers(min_value=0, max_value=2), st.floats(min_value=0., max_value=360.))
    def test_coord_tr_1(self, i, a):
        assert test_coord_tr_1(i, a)


def test_grid_origo_1(dx, dy, dz):
    d = np.array([dx, dy, dz])
    size = Lx, Ly, Lz = 80, 60, 20
    shape = nx, ny, nz = 8, 6, 2
    o1 = np.array([0., 0., 0.])
    coords1, topo1 = grid(size=size, shape=shape, eshape='H8', shift=o1)
    o2 = o1 + d
    coords2, topo2 = grid(size=size, shape=shape, eshape='H8', shift=o2)
    topo2 += coords1.shape[0]
    A = CartesianFrame(dim=3)
    pd1 = PolyData(coords=coords1, topo=topo1, frame=A)
    pd2 = PolyData(coords=coords2, topo=topo2, frame=A)
    assert np.max(np.abs(pd2.center() - pd1.center() - d)) < 1e-8
    return True


def test_volume_H8_1(size, shape):
    coords, topo = grid(size=size, shape=shape, eshape='H8')
    A = CartesianFrame(dim=3)
    pd = PolyData(coords=coords, topo=topo, frame=A)
    Lx, Ly, Lz = size
    V = Lx * Ly * Lz
    if not np.max(np.abs(V - pd.volume())) < 1e-8:
        return False
    return True


def test_volume_H8_2(size, shape):
    coords1, topo1 = grid(size=size, shape=shape, eshape='H8')
    coords2, topo2 = grid(size=size, shape=shape, eshape='H8')
    coords = np.vstack([coords1, coords2])
    topo2 += coords1.shape[0]

    A = CartesianFrame(dim=3)
    pd = PolyData(coords=coords, frame=A)
    pd['group1']['mesh1'] = PolyData(topo=topo1)
    pd['group2', 'mesh2'] = PolyData(topo=topo2)

    Lx, Ly, Lz = size
    V = Lx * Ly * Lz * 2
    if not np.max(np.abs(V - pd.volume())) < 1e-8:
        return False
    return True


def test_volume_TET4_1(size, shape):
    A = CartesianFrame(dim=3)
    mesh2d = TriMesh(size=size[:2], shape=shape[:2], frame=A)
    mesh3d = mesh2d.extrude(h=size[2], N=shape[2])
    Lx, Ly, Lz = size
    V = Lx * Ly * Lz
    if not np.max(np.abs(V - mesh3d.volume())) < 1e-8:
        return False
    return True


if __name__ == "__main__":

    assert test_grid_origo_1(1., 1., 1.)
    assert test_coord_tr_1(2, 30)

    A = CartesianFrame(dim=3)
    mesh = TriMesh(size=(800, 600), shape=(10, 10), frame=A)
   
    assert test_volume_TET4_1(size=(1, 1, 1), shape=(2, 2, 2))
    
    assert test_volume_H8_1(size=(1, 1, 1), shape=(2, 2, 2))
    
    unittest.main()