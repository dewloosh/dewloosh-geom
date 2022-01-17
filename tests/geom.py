# -*- coding: utf-8 -*-
from dewloosh.geom.space import PointCloud
from dewloosh.geom.space import StandardFrame
from dewloosh.math.linalg import Vector
from dewloosh.geom import TriMesh
from dewloosh.geom.tri.triang import circular_disk
from dewloosh.geom.polydata import PolyData
from dewloosh.geom.rgrid import grid
import numpy as np
from numba import njit
from hypothesis import given, strategies as st
import unittest


def test_coord_tr_1(i, a):
    A = StandardFrame(dim=3)
    coords = PointCloud([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ], frame=A)
    amounts = [0, 0, 0]
    amounts[i] = a*np.pi/180
    B = A.orient_new('Body', amounts, 'XYZ')
    arr_new = Vector(coords.array, frame=A).view(B)
    coords_new = Vector(arr_new, frame=B)
    return np.max(np.abs(coords_new.view() - coords.view())) < 1e-8


@njit
def get_inds(arr):
    return arr.inds

def test_inds_1():
    inds = np.array([0, 1, 2])
    COORD = PointCloud([
        [0, 0, 0], 
        [0, 0, 1.], 
        [0, 0, 0]
        ], inds=inds)
    c1 = np.array_equal(COORD[1:].inds, inds[1:])
    c2 = np.array_equal(get_inds(COORD[1:]), inds[1:])
    return c1 and c2



class TestCoords(unittest.TestCase):
                   
    @given(st.integers(min_value=0, max_value=2), st.floats(min_value=0., max_value=360.))
    def test_coord_tr_1(self, i, a):
        assert test_coord_tr_1(i, a)
        
    def test_coord_inds_1(self):
        assert test_inds_1()
        
        
def test_triang_1():
    try:
        A = StandardFrame(dim=3)    
        mesh = TriMesh(size=(800, 600), shape=(10, 10), frame=A)
        assert np.isclose(mesh.area(), 800*600)        
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e
    

def test_circular_disk_1(min_radius, max_radius, n_angles, n_radii):
    try:    
        points, triangles = \
            circular_disk(n_angles, n_radii, min_radius, max_radius)
        mesh = TriMesh(points=points, triangles=triangles, frame=A)
        a = np.pi * (max_radius**2 - min_radius**2)
        assert np.isclose(mesh.area(), a, atol=0, rtol=a/1000)
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e
    
    
def test_grid_origo_1(dx, dy, dz):
    d = np.array([dx, dy, dz])
    size = Lx, Ly, Lz = 80, 60, 20
    shape = nx, ny, nz = 8, 6, 2
    o1 = np.array([0., 0., 0.])
    coords1, topo1 = grid(size=size, shape=shape, eshape='H8', shift=o1)
    o2 = o1 + d
    coords2, topo2 = grid(size=size, shape=shape, eshape='H8', shift=o2)
    topo2 += coords1.shape[0]
    A = StandardFrame(dim=3)  
    coords = np.vstack([coords1, coords2])
    pd1 = PolyData(coords=coords1, topo=topo1, frame=A)
    pd2 = PolyData(coords=coords2, topo=topo2, frame=A)
    assert np.max(np.abs(pd2.center() - pd1.center() - d)) < 1e-8
    return True
        
    
def test_grid_volume_1(size, shape):
    coords1, topo1 = grid(size=size, shape=shape, eshape='H8')
    coords2, topo2 = grid(size=size, shape=shape, eshape='H8')
    coords = np.vstack([coords1, coords2])
    topo2 += coords1.shape[0]
    
    A = StandardFrame(dim=3)  
    pd = PolyData(coords=coords, frame=A)
    pd['group1']['mesh1'] = PolyData(topo=topo1)
    pd['group2', 'mesh2'] = PolyData(topo=topo2)
    
    Lx, Ly, Lz = size
    V = Lx * Ly * Lz * 2
    if not np.max(np.abs(V - pd.volume())) < 1e-8:
        return False   
    return True

    
class TestTriang(unittest.TestCase):
                   
    def test_triangulate_1(self):
        assert test_triang_1()
    
    @given(
        st.floats(min_value=1., max_value=5.), 
        st.floats(min_value=6., max_value=10.),
        st.integers(min_value=100, max_value=150),
        st.integers(min_value=60, max_value=100)
        )
    def test_circular_disk_1(self, min_radius, max_radius, n_angles, n_radii):
        assert test_circular_disk_1(min_radius, max_radius, n_angles, n_radii)
        
      
            
if __name__ == "__main__":  
    
    assert test_grid_origo_1(1., 1., 1.)
    assert test_coord_tr_1(2, 30)
    
    A = StandardFrame(dim=3)
    mesh = TriMesh(size=(800, 600), shape=(10, 10), frame=A)
    
    assert test_triang_1()
    
    unittest.main()