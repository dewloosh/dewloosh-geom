# -*- coding: utf-8 -*-
import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck
import unittest

from dewloosh.geom import TriMesh, CartesianFrame
from dewloosh.geom.tri.triang import circular_disk
from dewloosh.geom.cells import T3, T6


settings.register_profile(
    "tri_test",
    max_examples=100,
    deadline=None,  # ms
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)


def test_area_T3(Lx, Ly, nx, ny):
    try:
        A = CartesianFrame(dim=3)
        mesh = TriMesh(size=(Lx, Ly), shape=(nx, ny), frame=A, celltype=T3)
        assert np.isclose(mesh.area(), Lx*Ly)
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e
    

def test_area_T6(Lx, Ly, nx, ny):
    try:
        A = CartesianFrame(dim=3)
        mesh = TriMesh(size=(Lx, Ly), shape=(nx, ny), frame=A, celltype=T6)
        assert np.isclose(mesh.area(), Lx*Ly)
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e


def test_area_circular_disk_T3(min_radius, max_radius, n_angles, n_radii):
    try:
        A = CartesianFrame(dim=3)
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
    

def test_area_circular_disk_T6(min_radius, max_radius, n_angles, n_radii):
    try:
        A = CartesianFrame(dim=3)
        points, triangles = \
            circular_disk(n_angles, n_radii, min_radius, max_radius)
        mesh = TriMesh(points=points, triangles=triangles, frame=A, celltype=T6)
        a = np.pi * (max_radius**2 - min_radius**2)
        assert np.isclose(mesh.area(), a, atol=0, rtol=a/1000)
        return True
    except AssertionError:
        return False
    except Exception as e:
        raise e


class TestTri(unittest.TestCase):

    @given(
        st.floats(min_value=1., max_value=10.),
        st.floats(min_value=1., max_value=10.),
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=2, max_value=10)
    )
    @settings(settings.load_profile("tri_test"))
    def test_area_T3(self, Lx, Ly, nx, ny):
        assert test_area_T3(Lx, Ly, nx, ny)
        
    @given(
        st.floats(min_value=1., max_value=10.),
        st.floats(min_value=1., max_value=10.),
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=2, max_value=10)
    )
    @settings(settings.load_profile("tri_test"))
    def test_area_T6(self, Lx, Ly, nx, ny):
        assert test_area_T6(Lx, Ly, nx, ny)

    @given(
        st.floats(min_value=1., max_value=5.),
        st.floats(min_value=6., max_value=10.),
        st.integers(min_value=100, max_value=150),
        st.integers(min_value=60, max_value=100)
    )
    @settings(settings.load_profile("tri_test"))
    def test_area_circular_disk_T3(self, min_radius, max_radius, n_angles, n_radii):
        assert test_area_circular_disk_T3(min_radius, max_radius, n_angles, n_radii)
        
    @given(
        st.floats(min_value=1., max_value=5.),
        st.floats(min_value=6., max_value=10.),
        st.integers(min_value=100, max_value=150),
        st.integers(min_value=60, max_value=100)
    )
    @settings(settings.load_profile("tri_test"))
    def test_area_circular_disk_T6(self, min_radius, max_radius, n_angles, n_radii):
        assert test_area_circular_disk_T6(min_radius, max_radius, n_angles, n_radii)


if __name__ == "__main__":
            
    unittest.main()
