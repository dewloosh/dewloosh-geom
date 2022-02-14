# -*- coding: utf-8 -*-
import numpy as np
from hypothesis import given, strategies as st
import unittest

from dewloosh.geom import PointCloud, triangulate


def test_pointcloud_basic():
    coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
    coords = PointCloud(coords)
    coords.center()
    coords.centralize()
    d = np.array([1., 0., 0.])
    coords.rotate('Body', [0, 0, np.pi/2], 'XYZ').move(d)
    return np.all(np.isclose(coords.center(), d))


def test_pointcloud_path_1():
    """
    A triangulation travels a self-closing cycle and should return to
    itself (DCM matrix of the frame must be the identity matrix). 
    """
    coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
    coords = PointCloud(coords)
    coords.centralize()
    old = coords.show()
    d = np.array([1., 0., 0.])
    r = 'Body', [0, 0, np.pi/2], 'XYZ'
    coords.move(d, coords.frame).rotate(*r).move(d, coords.frame).\
        rotate(*r).move(d, coords.frame).rotate(*r).move(d, coords.frame).rotate(*r)
    new = coords.show()
    hyp_1 = np.all(np.isclose(old, new))
    hyp_2 = np.all(np.isclose(np.eye(3), coords.frame.dcm()))
    return hyp_1 & hyp_2


class TestPointCloud(unittest.TestCase):
                   
    #@given(st.integers(min_value=0, max_value=2), st.floats(min_value=0., max_value=360.))
    #def test_coord_tr_1(self, i, a):
    #    assert test_coord_tr_1(i, a)
        
    def test_pointcloud_basic(self):
        assert test_pointcloud_basic()
        
    def test_pointcloud_path_1(self):
        assert test_pointcloud_path_1()
        
                    
if __name__ == "__main__":  
    
    assert test_pointcloud_basic()
    assert test_pointcloud_path_1()
    
    unittest.main()