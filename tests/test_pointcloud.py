# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dewloosh.geom import PointCloud, triangulate, CartesianFrame


class TestPointCloud(unittest.TestCase):
                           
    def test_pointcloud_basic(self):
        def test_pointcloud_basic():
            coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
            coords = PointCloud(coords)
            coords.center()
            coords.centralize()
            d = np.array([1., 0., 0.])
            coords.rotate('Body', [0, 0, np.pi/2], 'XYZ').move(d)
            return np.all(np.isclose(coords.center(), d))
        assert test_pointcloud_basic()
        
    def test_pointcloud_path_1(self):
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
        assert test_pointcloud_path_1()
        
    def test_pointcloud_1(self):
        c = np.array([[0, 0, 0], [0, 0, 1.], [0, 0, 0]])
        COORD = PointCloud(c, inds=np.array([0, 1, 2, 3]))
        COORD[:, :].inds
        arr1 = COORD[1:]
        arr1.inds
        arr2 = arr1[1:]
        arr2.inds
        COORD.to_numpy()
        COORD @ np.eye(3)
        coords, topo, _ = triangulate(size=(800, 600), shape=(10, 10))
        coords = PointCloud(coords)
        coords.center()
        coords.centralize()
        coords.center()
        frameA = CartesianFrame(axes=np.eye(3), origo=np.array([-500., 0., 0.]))
        frameB = CartesianFrame(axes=np.eye(3), origo=np.array([+500., 0., 0.])) 
        frameA.origo()
        frameB.origo()
        coords.center()
        coords.center(frameA)
        coords.center(frameB)
        
                    
if __name__ == "__main__":  
        
    unittest.main()