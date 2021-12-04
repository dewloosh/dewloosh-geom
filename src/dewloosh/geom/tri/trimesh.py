# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
import matplotlib.tri as tri
import scipy.spatial
from scipy.spatial.qhull import Delaunay as spDelaunay
from dewloosh.geom.topo import unique_topo_data
from dewloosh.geom.tri.triutils import edges_tri
from vtk import vtkIdList

__all__ = ['triangulate']


def triangulate(*args, points = None, size : tuple = None,
                shape : tuple = None, origo : tuple = None,
                backend = 'mpl', random = False, triangles = None,
                triobj = None, return_lines = False, **kwargs):

    if len(args) > 0:
        if is_triobj(args[0]):
            triobj = args[0]
    if triobj is not None:
        points, triangles = get_triobj_data(triobj, *args, **kwargs)
    else:
        # create points from input
        if points is None:
            assert size is not None, \
                "Either a collection of points, or the size of a " \
                "rectangular domain must be provided!"
            if origo is None:
                origo = (0, 0, 0)
            else:
                if len(origo) == 2:
                    origo = origo + (0,)
            if shape is None:
                shape = (1, 1)
            if isinstance(shape, int):
                if random:
                    x = np.hstack([np.array([0,1,1,0], dtype = np.float32),
                                   np.random.rand(shape)])
                    y = np.hstack([np.array([0,0,1,1], dtype = np.float32),
                                   np.random.rand(shape)])
                    z = np.zeros(len(x), dtype = np.float32)
                    points = np.c_[x * size[0] - origo[0],
                                   y * size[1] - origo[1],
                                   z - origo[2]]
                else:
                    size = (shape, shape)
            if isinstance(size, tuple):
                x = np.linspace(-origo[0], size[0]-origo[0],
                                num=shape[0])
                y = np.linspace(-origo[1], size[1]-origo[1],
                                num=shape[1])
                z = np.zeros(len(x), dtype = np.float32) - origo[2]
                xx, yy = np.meshgrid(x, y)
                zz = np.zeros(xx.shape, dtype = xx.dtype)
                # Get the points as a 2D NumPy array (N by 2)
                points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]

        # generate triangles
        if triangles is None:
            if backend == 'mpl':
                triobj = tri.Triangulation(points[:, 0], points[:, 1])
                triangles = triobj.triangles
            elif backend == 'scipy':
                triobj = scipy.spatial.Delaunay(points[:, 0:2])
                triangles = triobj.vertices
            elif backend == 'pv':
                cloud = pv.PolyData(points)
                triobj = cloud.delaunay_2d()
                nCell = triobj.n_cells
                triangles = np.zeros((nCell, 3), dtype = np.int32)
                for cellID in range(nCell):
                    idlist = vtkIdList()
                    triobj.GetCellPoints(cellID, idlist)
                    n = idlist.GetNumberOfIds()
                    triangles[cellID] = [idlist.GetId(i) for i in range(n)]
        else:
            assert backend == 'mpl', "This feature is not yet supported by " \
                "other backends, only matplotlib."
            triobj = tri.Triangulation(points[:, 0], points[:, 1],
                                       triangles = triangles)
    if return_lines:
        edges, edgeIDs = unique_topo_data(edges_tri(triangles))
        return points, edges, triangles, edgeIDs, triobj
    return points, triangles, triobj


def triobj_to_mpl(triobj, *args, **kwargs):
    """
    Transforms a triangulation object into a matplotlib.tri.Triangulation
    object.
    """
    assert is_triobj(triobj)
    if isinstance(triobj, tri.Triangulation):
        return triobj
    else:
        points, triangles = get_triobj_data(triobj, *args, **kwargs)
        kwargs['backend'] = 'mpl'
        _, _, triang = triangulate(*args, points = points,
                                   triangles = triangles, **kwargs)
        return triang


def get_triobj_data(obj = None, *args, trim2d = True, **kwarg):
    coords, topo = None, None
    if isinstance(obj, spDelaunay):
        coords = obj.points
        topo = obj.vertices
    elif isinstance(obj, tri.Triangulation):
        coords = np.vstack((obj.x, obj.y)).T
        topo = obj.triangles
    elif isinstance(obj, pv.PolyData):
        if trim2d:
            coords = obj.points[:, 0:2]
        else:
            coords = obj.points
        triang = obj.delaunay_2d()
        nCell = triang.n_cells
        topo = np.zeros((nCell, 3), dtype = np.int32)
        for cellID in range(nCell):
            idlist = vtkIdList()
            triang.GetCellPoints(cellID, idlist)
            n = idlist.GetNumberOfIds()
            topo[cellID] = [idlist.GetId(i) for i in range(n)]
    if coords is None or topo is None:
        raise RuntimeError('Failed to recognize a valid triangulation, '
                           'look for improper input.')
    return coords, topo


def is_triobj(triobj):
    try:
        if isinstance(triobj, spDelaunay) or \
                isinstance(triobj, tri.Triangulation):
            return True
        elif isinstance(triobj, pv.PolyData):
            if hasattr(triobj, 'delaunay_2d'):
                return True
    except Exception:
        return False


if __name__ == '__main__':
    
    points, triangles, triobj = triangulate(size = (800, 600), 
                                            shape = (10, 10))