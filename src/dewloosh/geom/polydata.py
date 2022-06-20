# -*- coding: utf-8 -*-
from typing import Iterable
from copy import copy
from typing import Union, Hashable
from numpy import ndarray
import numpy as np

from dewloosh.core import DeepDict
from dewloosh.math.linalg import Vector, ReferenceFrame as FrameLike
from dewloosh.math.linalg.vector import VectorBase
from dewloosh.math.array import atleast3d, repeat

from .space import CartesianFrame
from .utils import cells_coords, cells_around, cell_center_bulk
from .topo import detach_mesh_bulk as detach_mesh
from .utils import k_nearest_neighbours as KNN
from .vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk
from .polygon import Triangle
from .cells.q4 import Q4 as Quadrilateral
from .polyhedron import HexaHedron, Wedge, TriquadraticHexaHedron
from .utils import index_of_closest_point, nodal_distribution_factors
from .topo import regularize, nodal_adjacency, cells_at_nodes
from .space import PointCloud
from .topo.topologyarray import TopologyArray
from .pointdata import PointData
from .celldata import CellData

from .config import __hasvtk__, __haspyvista__
if __hasvtk__:
    import vtk
if __haspyvista__:
    import pyvista as pv


def gen_frame(coords): return CartesianFrame(dim=coords.shape[1])


VectorLike = Union[Vector, ndarray]


def find_pointdata_in_args(*args):
    list(filter(lambda i: isinstance(i, PointData)))


__all__ = ['PointData']


class PolyData(DeepDict):
    """
    A class to handle complex polygonal meshes.

    The `PolyData` class is arguably the most important class
    in the geometry submodule. It manages polygons of different
    kinds as well their data in a memory efficient way.

    The implementation is based on the `awkward` DeepDict, which provides 
    memory-efficient, numba-jittable data classes to deal with dense, sparse, 
    complete or incomplete data. These data structures are managed in pure
    Python by the `DeepDict` class.

    Examples
    --------
    >>> from dewloosh.geom import PolyData
    >>> from dewloosh.geom.rgrid import grid
    >>> size = Lx, Ly, Lz = 100, 100, 100
    >>> shape = nx, ny, nz = 10, 10, 10
    >>> coords, topo = grid(size=size, shape=shape, eshape='H27')
    >>> pd = PolyData(coords=coords)
    >>> pd['A']['Part1'] = PolyData(topo=topo[:10])
    >>> pd['B']['Part2'] = PolyData(topo=topo[10:-10])
    >>> pd['C']['Part3'] = PolyData(topo=topo[-10:])
    
    """

    _point_array_class_ = PointCloud
    _point_class_ = PointData
    _cell_classes_ = {
        8: HexaHedron,
        6: Wedge,
        4: Quadrilateral,
        3: Triangle,
        27: TriquadraticHexaHedron,
    }

    def __init__(self, pd=None, cd=None, *args, coords=None, topo=None, 
                 celltype=None, frame: FrameLike = None, newaxis: int = 2, 
                 cell_fields=None, point_fields=None, **kwargs):          
        self.pointdata = None
        self.celldata = None
        if isinstance(pd, PointData):
            self.pointdata = pd
            if isinstance(cd, CellData):
                self.celldata = cd
        elif isinstance(pd, CellData):
            self.celldata = pd 
            if isinstance(cd, PointData):
                self.pointdata = cd
        self.celltype = None
        self.cell_index_manager = None
        self._frame = frame
        
        super().__init__(*args, **kwargs)

        if self.pointdata is None and coords is not None:    
            point_fields = {} if point_fields is None else point_fields
            pointtype = self.__class__._point_class_
            assert self.is_root(), "Currently only the top-level structure \
                (root) can hold onto point-like data."
            self.pointdata = pointtype(coords=coords, frame=frame, 
                                       newaxis=newaxis, stateful=True, 
                                       fields=point_fields)
            self.cell_index_manager = IndexManager()

        
        if self.celldata is None and topo is not None:
            cell_fields = {} if cell_fields is None else cell_fields
            if celltype is None:
                celltype = self.__class__._cell_classes_.get(
                    topo.shape[1], None)
            if not issubclass(celltype, CellData):
                raise TypeError("Invalid cell type <{}>".format(celltype))
            #assert celltype is not None

            root = self.root()
            pd=root.pointdata
            if isinstance(topo, np.ndarray):
                topo = topo.astype(int)
            else:
                raise TypeError("Topo must be an 1d array of integers.")

            if root.cell_index_manager is not None:
                GIDs = np.array(
                    root.cell_index_manager.generate(topo.shape[0]))
                cell_fields['id'] = GIDs            
            self.celldata = celltype(topo, fields=cell_fields, pointdata=pd)
            self.celltype = celltype

    def blocks(self, *args, inclusive=False, blocktype=None, deep=True, **kwargs):
        dtype = PolyData if blocktype is None else blocktype
        return self.containers(self, inclusive=inclusive, dtype=dtype, deep=deep)

    def cellblocks(self, *args, **kwargs):
        return filter(lambda i: i.celldata is not None, self.blocks(*args, **kwargs))

    @property
    def frame(self):
        if self.is_root():
            return self._frame
        else:
            f = self._frame
            return f if f is not None else self.parent.frame

    @property
    def frames(self):
        if self.celldata is not None and 'frames' in self.celldata.fields:
            return self.celldata.frames.to_numpy()

    @frames.setter
    def frames(self, value):
        assert self.celldata is not None
        if isinstance(value, ndarray):
            value = atleast3d(value)
            if len(value) == 1:
                value = repeat(value[0], len(self.celldata._wrapped))
            else:
                assert len(value) == len(self.celldata._wrapped)
            self.celldata._wrapped['frames'] = value
        else:
            raise TypeError(('Type {} is not a supported' +
                             ' type to specify frames.').format(type(value)))

    def points(self, *args, return_inds=False, **kwargs) -> PointCloud:
        if self.is_root():
            coords = self.pointdata.x.to_numpy()
            frame = self.frame
            frame = gen_frame(coords) if frame is None else frame
            return self.__class__._point_array_class_(coords, frame=frame)
        else:
            # returns a sorted array of unique indices
            inds = np.unique(self.topology())
            if return_inds:
                return self.root().points()[inds, :], inds
            else:
                return self.root().points()[inds, :]

    def coords(self, *args, return_inds=False, **kwargs) -> VectorBase:
        if self.is_root():
            return self.points().show()
        else:
            # returns a sorted array of unique indices
            inds = np.unique(self.topology())
            if return_inds:
                return self.root().coords()[inds, :], inds
            else:
                return self.root().coords()[inds, :]

    def move(self, v: VectorLike, frame: FrameLike = None):
        if self.is_root():
            pc = self.points()
            pc.move(v, frame)
            self.pointdata['x'] = pc.array
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            pc.move(v, frame)
            root.pointdata['x'] = pc.array
        return self

    def rotate(self, *args, **kwargs):
        if self.is_root():
            pc = self.points()
            pc.rotate(*args, **kwargs)
            self.pointdata['x'] = pc.show(self.frame)
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            pc.rotate(*args, **kwargs)
            root.pointdata['x'] = pc.show(self.frame)
        return self

    def topology(self, *args, return_inds=False, **kwargs):
        blocks = list(self.cellblocks(*args, inclusive=True, **kwargs))
        topo = list(map(lambda i: i.celldata.nodes.to_numpy(), blocks))
        shapes = np.array(list(map(lambda arr: arr.shape[1], topo)))
        if not np.all(shapes == shapes[0]):
            if return_inds:
                raise NotImplementedError
            return TopologyArray(*topo)
        else:
            topo = np.vstack(topo)
            if return_inds:
                inds = list(map(lambda i: i.celldata.id.to_numpy(), blocks))
                return topo, np.concatenate(inds)
            else:
                return topo

    def cells_at_nodes(self, *args, **kwargs):
        topo = self.topology()
        return cells_at_nodes(topo, *args, **kwargs)

    def cells_around_cells(self, radius=None, frmt='dict'):
        if radius is None:
            # topology based
            raise NotImplementedError
        else:
            return cells_around(self.centers(), radius, frmt=frmt)

    def nodal_adjacency_matrix(self, *args, **kwargs):
        topo = self.topology()
        return nodal_adjacency(topo, *args, **kwargs)

    def number_of_cells(self):
        blocks = self.cellblocks(inclusive=True)
        return np.sum(list(map(lambda i: len(i.celldata), blocks)))

    def number_of_points(self):
        return len(self.root().pointdata)

    def cells_coords(self, *args, _topo=None, **kwargs):
        _topo = self.topology() if _topo is None else _topo
        return cells_coords(self.root().coords(), _topo)

    def center(self, target: FrameLike = None):
        if self.is_root():
            return self.points().center(target)
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            return pc.center(target)

    def centers(self, *args, target: FrameLike = None, **kwargs):
        if self.is_root():
            coords = self.points().show(target)
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            coords = pc.show(target)
        return cell_center_bulk(coords, self.topology(*args, **kwargs))

    def centralize(self, target: FrameLike = None):
        pc = self.root().points()
        pc.centralize(target)
        self.pointdata['x'] = pc.show(self.frame)
        return self

    def k_nearest_cell_neighbours(self, k, *args, knn_options=None, **kwargs):
        """
        Returns the k closest neighbours of the cells of the mesh, based
        on the centers of each cell.

        The argument `knn_options` is passed to the KNN search algorithm,
        the rest to the `centers` function of the mesh.
        """
        c = self.centers(*args, **kwargs)
        knn_options = {} if knn_options is None else knn_options
        return KNN(c, c, k=k, **knn_options)

    def areas(self, *args, **kwargs):
        coords = self.root().coords()
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        blocks2d = filter(lambda b: b.celltype.NDIM < 3, blocks)
        amap = map(lambda b: b.celldata.areas(coords=coords), blocks2d)
        return np.concatenate(list(amap))

    def area(self, *args, **kwargs):
        return np.sum(self.areas(*args, **kwargs))

    def volumes(self, *args, **kwargs):
        coords = self.root().coords()
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        vmap = map(lambda b: b.celldata.volumes(coords=coords), blocks)
        return np.concatenate(list(vmap))

    def volume(self, *args, **kwargs):
        return np.sum(self.volumes(*args, **kwargs))

    def index_of_closest_point(self, target, *args, **kwargs):
        return index_of_closest_point(self.coords(), target)

    def index_of_closest_cell(self, target, *args, **kwargs):
        return index_of_closest_point(self.centers(), target)

    def set_nodal_distribution_factors(self, *args, **kwargs):
        self.nodal_distribution_factors(*args, store=True, **kwargs)

    def nodal_distribution_factors(self, *args, assume_regular=False,
                                   key='ndf', store=False, measure='volume',
                                   load=None, weights=None, **kwargs):
        if load is not None:
            if isinstance(load, str):
                blocks = self.cellblocks(inclusive=True)
                def foo(b): return b.celldata._wrapped[load].to_numpy()
                return np.vstack(list(map(foo, blocks)))

        topo, inds = self.topology(return_inds=True)

        if measure == 'volume':
            weights = self.volumes()
        elif measure == 'uniform':
            weights = np.ones(topo.shape[0], dtype=float)

        argsort = np.argsort(inds)
        topo = topo[argsort]
        weights = weights[argsort]
        if not assume_regular:
            topo, _ = regularize(topo)
        factors = nodal_distribution_factors(topo, weights)
        if store:
            blocks = self.cellblocks(inclusive=True)
            def foo(b): return b.celldata.set_nodal_distribution_factors(
                factors, key=key)
            list(map(foo, blocks))
        return factors

    def to_vtk(self, deepcopy=True, fuse=True):
        if not __hasvtk__:
            raise ImportError
        coords = self.coords()
        blocks = list(self.cellblocks(inclusive=True))
        if fuse:
            if len(blocks) == 1:
                topo = blocks[0].celldata.nodes.to_numpy().astype(np.int64)
                ugrid = mesh_to_vtk(*detach_mesh(coords, topo),
                                    blocks[0].celltype.vtkCellType, deepcopy)
                return ugrid
            mb = vtk.vtkMultiBlockDataSet()
            mb.SetNumberOfBlocks(len(blocks))
            for i, block in enumerate(blocks):
                topo = block.celldata.nodes.to_numpy().astype(np.int64)
                ugrid = mesh_to_vtk(*detach_mesh(coords, topo),
                                    block.celltype.vtkCellType, deepcopy)
                mb.SetBlock(i, ugrid)
            return mb
        else:
            res = []
            for i, block in enumerate(blocks):
                topo = block.celldata.nodes.to_numpy()
                ugrid = mesh_to_vtk(*detach_mesh(coords, topo),
                                    block.celltype.vtkCellType, deepcopy)
                res.append(ugrid)
            return res

    def to_pv(self, *args, fuse=True, **kwargs):
        if not __haspyvista__:
            raise ImportError
        if fuse:
            multiblock = pv.wrap(self.to_vtk(*args, fuse=True, **kwargs))
            try:
                multiblock.wrap_nested()
            except AttributeError:
                pass
            return multiblock
        else:
            return [pv.wrap(i) for i in self.to_vtk(*args, fuse=False, **kwargs)]

    def plot(self, *args, deepcopy=True, jupyter_backend='pythreejs',
             show_edges=True, notebook=False, theme='document', **kwargs):
        if not __haspyvista__:
            raise ImportError
        if theme is not None:
            pv.set_plot_theme(theme)
        poly = self.to_pv(deepcopy=deepcopy)
        if notebook:
            poly.plot(*args, jupyter_backend=jupyter_backend,
                      show_edges=show_edges, notebook=notebook, **kwargs)
        else:
            poly.plot(*args, show_edges=show_edges, notebook=notebook,
                      **kwargs)

    def __join_parent__(self, parent: DeepDict, key: Hashable = None):
        super().__join_parent__(parent, key)
        if self._root.cell_index_manager is not None and self.celldata is not None:
            GIDs = np.array(
                self._root.cell_index_manager.generate(len(self.celldata)))
            self.celldata['id'] = GIDs

    def __repr__(self):
        return 'PolyData(%s)' % (dict.__repr__(self))


class IndexManager(object):
    """This object ought to guarantee, that every cell in a 
    model has a unique ID."""

    def __init__(self, start=0):
        self.queue = []
        self.next = start

    def generate(self, n=1):
        nQ = len(self.queue)
        if nQ > 0:
            if n == 1:
                res = self.queue.pop()
            else:
                if nQ >= n:
                    res = self.queue[:n]
                    del self.queue[:n]
                else:
                    res = copy(self.queue)
                    res.extend(range(self.next, self.next + n - nQ))
                    self.queue = []
                self.next += n - nQ
        else:
            if n == 1:
                res = self.next
            else:
                res = list(range(self.next, self.next + n))
            self.next += n
        return res

    def recycle(self, *args, **kwargs):
        for a in args:
            if isinstance(a, Iterable):
                self.queue.extend(a)
            else:
                self.queue.append(a)