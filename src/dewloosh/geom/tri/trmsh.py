# -*- coding: utf-8 -*-
from dewloosh.geom.polydata import PolyData
from dewloosh.geom.polyhedron import TetraHedron as Tetra


__all__ = ['TriMesh']


class TriMesh(PolyData):
    
    def __init__(self, *args,  points=None, triangles=None, **kwargs):
        # parent class handles pointdata and celldata creation
        points = points if points is not None else \
            kwargs.get('coords', None)
        triangles = triangles if triangles is not None else \
            kwargs.get('topo', None)
        if triangles is None:
            # this is here to avoid circular imports
            from dewloosh.geom.tri.trimesh import triangulate
            points, triangles, _ = triangulate(*args, points=points, **kwargs)
        super().__init__(*args, coords=points, topo=triangles, **kwargs)

    def extrude(self, amount, etype='TET', **mesh_params):
        """
        Exctrude mesh perpendicular to the plane of the triangulation.
        The target element type can be specified with the `etype` argument.
        """
        etype = Tetra if etype == 'TET' else etype
        raise NotImplementedError
    

if __name__ == '__main__':
    from dewloosh.geom import TriMesh
    trimesh = TriMesh(size=(800, 600), shape=(10, 10))
    trimesh.plot()