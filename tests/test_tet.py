# -*- coding: utf-8 -*-
import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck
import unittest

from dewloosh.geom import TriMesh, CartesianFrame
from dewloosh.geom.tri.triang import circular_disk
from dewloosh.geom.cells import T3, T6


settings.register_profile(
    "tet_test",
    max_examples=100,
    deadline=None,  # ms
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)


def test_vol_TET4(Lx, Ly, Lz, nx, ny, nz):
    try:
        A = CartesianFrame(dim=3)
        mesh2d = TriMesh(size=(Lx, Ly), shape=(nx, ny), frame=A, celltype=T3)
        mesh3d = mesh2d.extrude(h=Lz, N=nz)
        assert np.isclose(mesh3d.volume(), Lx*Ly*Lz)
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e
    
    
def test_vol_cylinder_TET4(min_radius, max_radius, height, 
                           n_angles, n_radii, n_z):
    try:
        A = CartesianFrame(dim=3)
        points, triangles = \
            circular_disk(n_angles, n_radii, min_radius, max_radius)
        mesh2d = TriMesh(points=points, triangles=triangles, frame=A)
        mesh3d = mesh2d.extrude(h=height, N=n_z)
        a = np.pi * (max_radius**2 - min_radius**2) * height
        assert np.isclose(mesh3d.volume(), a, atol=0, rtol=a/1000)
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e
    

class TestTet(unittest.TestCase):

    @given(
        st.floats(min_value=1., max_value=10.),
        st.floats(min_value=1., max_value=10.),
        st.floats(min_value=1., max_value=10.),
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=2, max_value=10)
    )
    @settings(settings.load_profile("tet_test"))
    def test_vol_TET4(self, Lx, Ly, Lz, nx, ny, nz):
        assert test_vol_TET4(Lx, Ly, Lz, nx, ny, nz)
        
    @given(
        st.floats(min_value=1., max_value=5.),
        st.floats(min_value=6., max_value=10.),
        st.floats(min_value=1., max_value=20.),
        st.integers(min_value=100, max_value=150),
        st.integers(min_value=60, max_value=100),
        st.integers(min_value=2, max_value=10)
    )
    @settings(settings.load_profile("tet_test"))
    def test_vol_cylinder_TET4(self, min_radius, max_radius, height, 
                               n_angles, n_radii, n_z):
        assert test_vol_cylinder_TET4(min_radius, max_radius, height, 
                                      n_angles, n_radii, n_z)


if __name__ == "__main__":
            
    unittest.main()
