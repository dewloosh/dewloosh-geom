# -*- coding: utf-8 -*-
from dewloosh.geom.polydata import PolyData
from dewloosh.geom.polyhedron import TetraHedron as Tetra
from dewloosh.geom.polygon import Triangle, QuadraticTriangle
from dewloosh.geom.space.utils import frames_of_surfaces as frames, \
    is_planar_surface as is_planar
from dewloosh.geom.tet.tetutils import extrude_T3_TET4
from dewloosh.geom.tri.triang import triangulate
from dewloosh.math.array import ascont
from dewloosh.geom.tri.triutils import edges_tri
from dewloosh.geom.topo import unique_topo_data
import numpy as np


class TriMesh(PolyData):
    """
    A class to handle triangular meshes.
    
    Examples
    --------
    Triangulate a rectangle of size 800x600 with a subdivision of 10x10
    and calculate the area
    
    >>> trimesh = TriMesh(size=(800, 600), shape=(10, 10))
    >>> trimesh.area()
    480000.0
    
    Extrude to create a tetrahedral mesh
    
    >>> tetmesh = trimesh.extrude(h=300, N=5)
    >>> tetmesh.volume()
    144000000.0
    
    Calculate normals and tell if the triangles form
    a planar surface or not
    
    >>> trimesh.normals()
    >>> trimesh.is_planar()
    True
    
    """
    
    def __init__(self, *args,  points=None, triangles=None, 
                 celltype=None, **kwargs):
        # parent class handles pointdata and celldata creation
        points = points if points is not None else \
            kwargs.get('coords', None)
        triangles = triangles if triangles is not None else \
            kwargs.get('topo', None)
        if triangles is None:
            try:
                points, triangles, _ = \
                    triangulate(*args, points=points, **kwargs)
            except Exception:
                raise RuntimeError
        if celltype is None and triangles is not None:
            if isinstance(triangles, np.ndarray):
                nNode = triangles.shape[1]
                if nNode == 3:
                    celltype = Triangle
                elif nNode == 6:
                    celltype = QuadraticTriangle
            else:
                raise NotImplementedError
        super().__init__(*args, coords=points, topo=triangles, 
                         celltype=celltype, **kwargs)
        
    def frames(self):
        """
        Returns the normalized coordinate frames of triangles.
        """
        return frames(self.coords(), self.topology())
        
    def normals(self):
        """
        Retuns the surface normals.
        """
        return ascont(self.frames()[:, 2, :])
        
    def is_planar(self) -> bool:
        """
        Returns `True` if the triangles form a planar surface.
        """
        return is_planar(self.normals())

    def extrude(self, *args, celltype=None, h=None, N=None, **kwargs):
        """
        Exctrude mesh perpendicular to the plane of the triangulation.
        The target element type can be specified with the `celltype` argument.
        
        Parameters
        ----------
        h : Float
            Size perpendicular to the plane of the surface to be extruded.
        
        N : Int
            Number of subdivisions along the perpendicular direction.
            
        Returns
        -------
        TetMesh
            A tetrahedral mesh.
        
        """
        from dewloosh.geom import TetMesh
        if not self.is_planar():
            raise RuntimeError("Only planar surfaces can be extruded!")
        assert celltype is None, "Currently only TET4 element is supported!"
        celltype = Tetra if celltype == None else celltype
        coords, topo = extrude_T3_TET4(self.coords(), self.topology(), h, N)
        return TetMesh(coords=coords, topo=topo, celltype=celltype)
    
    def edges(self, return_cells=False):
        """
        Returns point indices of the unique edges in the model.
        If `return_cells` is `True`, it also returns the edge
        indices of the triangles, referencing the edges.
        
        Parameters
        ----------
        return_cells : Bool, Optional
            If True, returns the edge indices of the triangles, 
            that can be used to reconstruct the topology. 
            Default is False.
        
        Returns
        -------
        numpy.ndarray
            Integer array of indices, representing point indices of edges.
            
        numpy.ndarray
            Integer array of indices, that together with the edge data 
            reconstructs the topology.
        """
        edges, edgeIDs = unique_topo_data(edges_tri(self.topology()))
        if return_cells:
            return edges, edgeIDs
        else:
            return edges


if __name__ == '__main__': 
    trimesh = TriMesh(size=(800, 600), shape=(10, 10))
    trimesh.plot()
    print(trimesh.area())
    tetmesh = trimesh.extrude(h=300, N=5)
    tetmesh.plot()
    print(tetmesh.volume())