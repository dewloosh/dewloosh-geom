# -*- coding: utf-8 -*-
from copy import copy, deepcopy
from typing import Union, Hashable, Collection, Iterable
from numpy import ndarray
import numpy as np
from awkward import Array as akarray

from dewloosh.core import DeepDict

from dewloosh.math.linalg.sparse import JaggedArray
from dewloosh.math.linalg import Vector, ReferenceFrame as FrameLike
from dewloosh.math.linalg.vector import VectorBase
from dewloosh.math.array import atleast3d, repeat, atleastnd

from dewloosh.geom.topo.topo import inds_to_invmap_as_dict, remap_topo_1d
from .space import CartesianFrame, PointCloud
from .utils import cells_coords, cells_around, cell_center_bulk
from .utils import k_nearest_neighbours as KNN
from .vtkutils import mesh_to_UnstructuredGrid as mesh_to_vtk
from .cells import T3 as Triangle, Q4 as Quadrilateral, H8 as Hexahedron, \
    H27 as TriquadraticHexaHedron, Q9, TET10
from .polyhedron import Wedge
from .utils import index_of_closest_point, nodal_distribution_factors
from .topo import regularize, nodal_adjacency, detach_mesh_bulk, \
    cells_at_nodes
from .topo.topoarray import TopologyArray
from .pointdata import PointData
from .celldata import CellData

from .config import __hasvtk__, __haspyvista__
if __hasvtk__:
    import vtk
if __haspyvista__:
    import pyvista as pv


VectorLike = Union[Vector, ndarray]
TopoLike = Union[ndarray, JaggedArray, akarray, TopologyArray]
NoneType = type(None)


def find_pointdata_in_args(*args):
    list(filter(lambda i: isinstance(i, PointData)))


__all__ = ['PointData']


class PolyData(DeepDict):
    """
    A class to handle complex polygonal meshes.

    The `PolyData` class is arguably the most important class
    in the geometry submodule. It manages polygons of different
    kinds as well their data in a memory efficient way.

    The implementation is based on the `awkward` library, which provides 
    memory-efficient, numba-jittable data classes to deal with dense, sparse, 
    complete or incomplete data. These data structures are managed in pure
    Python by the `DeepDict` class.

    Parameters
    ----------


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
    _frame_class_ = CartesianFrame
    _cell_class = NoneType
    _cell_classes_ = {
        8: Hexahedron,
        6: Wedge,
        4: Quadrilateral,
        9: Q9,
        10: TET10,
        3: Triangle,
        27: TriquadraticHexaHedron,
    }

    def __init__(self, pd=None, cd=None, *args, coords=None, topo=None,
                 celltype=None, frame: FrameLike = None, newaxis: int = 2,
                 cell_fields=None, point_fields=None, parent: DeepDict = None,
                 **kwargs):
        self._reset_point_data()
        self._reset_cell_data()
        self._frame = frame
        self._newaxis = newaxis
        self._parent = parent

        if isinstance(pd, PointData):
            self.pointdata = pd
            if isinstance(cd, CellData):
                self.celldata = cd
        elif isinstance(pd, CellData):
            self.celldata = pd
            if isinstance(cd, PointData):
                self.pointdata = cd

        super().__init__(*args, **kwargs)

        self.point_index_manager = IndexManager()
        self.cell_index_manager = IndexManager()

        if self.pointdata is None and coords is not None:
            point_fields = {} if point_fields is None else point_fields
            pointtype = self.__class__._point_class_
            GIDs = self.root().pim.generate_np(coords.shape[0])
            point_fields['id'] = GIDs
            self.pointdata = pointtype(coords=coords, frame=frame,
                                       newaxis=newaxis, stateful=True,
                                       fields=point_fields)

        if self.celldata is None and topo is not None:
            cell_fields = {} if cell_fields is None else cell_fields
            if celltype is None:
                celltype = self.__class__._cell_classes_.get(
                    topo.shape[1], None)
            if not issubclass(celltype, CellData):
                raise TypeError("Invalid cell type <{}>".format(celltype))

            root = self.root()
            pd = root.pointdata
            if isinstance(topo, np.ndarray):
                topo = topo.astype(int)
            else:
                raise TypeError("Topo must be an 1d array of integers.")

            GIDs = self.root().cim.generate_np(topo.shape[0])
            cell_fields['id'] = GIDs
            self.celldata = celltype(topo, fields=cell_fields, pointdata=pd)

        if self.celldata is not None:
            self.celltype = self.celldata.__class__

    @property
    def pim(self) -> 'IndexManager':
        return self.point_index_manager

    @property
    def cim(self) -> 'IndexManager':
        return self.cell_index_manager

    @property
    def parent(self) -> 'PolyData':
        return self._parent

    @parent.setter
    def parent(self, value: 'PolyData'):
        self._parent = value

    @property
    def address(self):
        if self.is_root():
            return []
        else:
            r = self.parent.address
            r.append(self.key)
            return r

    def is_source(self, key) -> bool:
        """
        Returns `True`, if the object is a valid source of data specified by `key`.

        Parameters
        ----------
        key : str
            A valid key to an `awkward` record.

        """
        key = 'x' if key is None else key
        return self.pointdata is not None and key in self.pointdata.fields

    def source(self, key=None) -> 'PolyData':
        """
        Returns the closest (going upwards in the hierarchy) block that holds on to data.
        If called without arguments, it is looking for a block with a valid pointcloud,
        definition, otherwise the field specified by the argument `key`.

        Parameters
        ----------
        key : str
            A valid key in any of the blocks with data. Default is None.

        """
        key = 'x' if key is None else key
        if self.pointdata is not None:
            if key in self.pointdata.fields:
                return self
        if not self.is_root():
            return self.parent.source(key=key)
        else:
            raise KeyError("No data found with key '{}'".format(key))

    def blocks(self, *args, inclusive=False, blocktype=None, deep=True,
               **kwargs) -> Collection['PolyData']:
        """
        Returns an iterable over inner dictionaries.
        """
        dtype = PolyData if blocktype is None else blocktype
        return self.containers(self, inclusive=inclusive, dtype=dtype, deep=deep)

    def pointblocks(self, *args, **kwargs) -> Collection['PolyData']:
        """
        Returns an iterable over blocks with point data.
        """
        return filter(lambda i: i.pointdata is not None, self.blocks(*args, **kwargs))

    def cellblocks(self, *args, **kwargs) -> Collection['PolyData']:
        """
        Returns an iterable over blocks with cell data.
        """
        return filter(lambda i: i.celldata is not None, self.blocks(*args, **kwargs))

    @property
    def pd(self):
        return self.pointdata
    
    @property
    def cd(self):
        return self.celldata
    
    @property
    def point_fields(self):
        """
        Returns the fields of all the pointdata of the object.

        Returns
        -------
        numpy.ndaray
            NumPy array of data keys.

        """
        pointblocks = list(self.pointblocks())
        m = map(lambda pb: pb.pointdata.fields, pointblocks)
        return np.unique(np.array(list(m)).flatten())

    @property
    def cell_fields(self):
        """
        Returns the fields of all the celldata of the object.

        Returns
        -------
        numpy.ndaray
            NumPy array of data keys.

        """
        cellblocks = list(self.cellblocks())
        m = map(lambda cb: cb.celldata.fields, cellblocks)
        return np.unique(np.array(list(m)).flatten())

    @property
    def frame(self) -> FrameLike:
        """Returns the frame of the points."""
        if self.is_root():
            if self._frame is not None:
                return self._frame
            else:
                dim = self.source().coords().shape[-1]
                self._frame = self._frame_class_(dim=dim)
                return self._frame
        else:
            f = self._frame
            return f if f is not None else self.parent.frame

    @property
    def frames(self) -> ndarray:
        """Returnes the frames of the cells."""
        if self.celldata is not None and 'frames' in self.celldata.fields:
            return self.celldata.frames.to_numpy()

    @frames.setter
    def frames(self, value: ndarray):
        """Sets the frames of the cells."""
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

    def _reset_point_data(self):
        self.pointdata = None
        self.cell_index_manager = None

    def _reset_cell_data(self):
        self.celldata = None
        self.celltype = None

    def rewire(self, deep=True):
        """
        Rewires topology according to the index mapping of the source object.

        Parameters
        ----------
        deep : bool
            If `True`, the action propagates down.

        Notes
        -----
        Unless node numbering was modified, subsequent executions have no effect
        after once called.

        """
        if not deep:
            if self.celldata is not None:
                s = self.source()
                if s is not self.root():
                    imap = self.source().pointdata['id'].to_numpy()
                    self.celldata.rewire(imap=imap)
        else:
            [c.rewire(deep=False) for c in self.cellblocks(inclusive=True)]

    def to_standard_form(self):
        """
        Transforms the problem to standard form.
        """
        if not self.is_root():
            raise NotImplementedError

        # merge points and point related data
        # + decorate the points with globally unique ids
        im = IndexManager()
        pointtype = self.__class__._point_class_
        pointblocks = list(self.pointblocks(inclusive=True))
        m = map(lambda pb: pb.pointdata.fields, pointblocks)
        fields = np.unique(np.array(list(m)).flatten())
        m = map(lambda pb: pb.pointdata.x.to_numpy(), pointblocks)
        X, frame, axis = np.vstack(list(m)), self._frame, self._newaxis
        if len(fields) > 0:
            point_fields = {}
            data = {f: [] for f in fields}
            for pb in pointblocks:
                GIDs = im.generate_np(len(pb.pointdata))
                pb.pointdata['id'] = GIDs
                for f in fields:
                    if f in pb.pointdata.fields:
                        data[f].append(pb.pointdata[f].to_numpy())
                    else:
                        data[f].append(np.zeros(len(pb.pointdata)))
            data.pop('x', None)
            for f in data.keys():
                nd = np.max([len(d.shape) for d in data[f]])
                fdata = list(map(lambda arr: atleastnd(arr, nd), data[f]))
                point_fields[f] = np.concatenate(fdata, axis=0)
        else:
            point_fields = None
        self.pointdata = pointtype(coords=X, frame=frame, newaxis=axis,
                                   stateful=True, fields=point_fields)

        # merge cells and cell related data
        # + rewire the topology based on the ids set in the previous block
        cellblocks = list(self.cellblocks(inclusive=True))
        m = map(lambda pb: pb.celldata.fields, cellblocks)
        fields = np.unique(np.array(list(m)).flatten())
        if len(fields) > 0:
            ndim = {f: [] for f in fields}
            for cb in cellblocks:
                imap = cb.source().pointdata['id'].to_numpy()
                # cb.celldata.rewire(imap=imap)  # this has been done at joining parent
                for f in fields:
                    if f in cb.celldata.fields:
                        ndim[f].append(len(cb.celldata[f].to_numpy().shape))
                    else:
                        cb.celldata[f] = np.zeros(len(cb.celldata))
                        ndim[f].append(1)
            ndim = {f: np.max(v) for f, v in ndim.items()}
            for cb in cellblocks:
                cb.celldata[f] = atleastnd(cb.celldata[f].to_numpy(), ndim[f])

        # free resources
        for pb in self.pointblocks(inclusive=False):
            pb._reset_point_data()

    def points(self, *args, return_inds=False, from_cells=False, **kwargs) -> PointCloud:
        """
        Returns the points as a `PointCloud` object.

        """
        frame = self.frame
        if from_cells:
            inds_ = np.unique(self.topology())
            x, inds = self.root().points(from_cells=False, return_inds=True)
            imap = inds_to_invmap_as_dict(inds)
            inds = remap_topo_1d(inds_, imap)
            coords, inds = x[inds, :], inds_
        else:
            # TODO : handle transformations here
            pb = list(self.pointblocks(inclusive=True))
            m = map(lambda pb: pb.pointdata.x.to_numpy(), pb)
            coords = np.vstack(list(m))
            m = map(lambda pb: pb.pointdata.id.to_numpy(), pb)
            inds = np.concatenate(list(m)).astype(int)
        __cls__ = self.__class__._point_array_class_
        points = __cls__(coords, frame=frame, inds=inds)
        if return_inds:
            return points, inds
        return points

    def coords(self, *args, return_inds=False, from_cells=False, **kwargs) -> VectorBase:
        """Returns the coordinates as an array."""
        if return_inds:
            p, inds = self.points(return_inds=True, from_cells=from_cells)
            return p.show(*args, **kwargs), inds
        else:
            return self.points(from_cells=from_cells).show(*args, **kwargs)

    def cells(self):
        """
        This should be the same to topology, what point is to coords,
        with no need to copy the underlying mechanism.

        The relationship of resulting object to the topology of a mesh should 
        be similar to that of `PointCloud` and the points in 3d space.

        """
        pass

    def topology(self, *args, return_inds=False, **kwargs):
        """
        Returns the topology as either a `numpy` or an `awkward` array.

        Notes
        -----
        The call automatically propagates down.

        """
        blocks = list(self.cellblocks(*args, inclusive=True, **kwargs))
        topo = list(map(lambda i: i.celldata.topology(), blocks))
        widths = np.concatenate(list(map(lambda t: t.widths(), topo)))
        jagged = not np.all(widths == widths[0])
        if jagged:
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

    def detach(self) -> 'PolyData':
        coords = self.root().coords(from_cells=False)
        pd = PolyData(coords=coords, frame=self.frame)
        l0 = len(self.address)
        for cb in self.cellblocks(inclusive=True):
            addr = cb.address
            if len(addr) > l0:
                cd = cb.celltype(
                    pointdata=pd, celldata=deepcopy(cb.celldata.db))
                pd[addr[l0:]] = PolyData(cd)
                assert pd[addr[l0:]].celldata is not None
        return pd

    def nummrg(self):
        pass

    def move(self, v: VectorLike, frame: FrameLike = None):
        """Moves the object."""
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
        """Rotates the object."""
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

    def cells_at_nodes(self, *args, **kwargs):
        """Returns the neighbouring cells of nodes."""
        topo = self.topology()
        return cells_at_nodes(topo, *args, **kwargs)

    def cells_around_cells(self, radius=None, frmt='dict'):
        """Returns the neares cells to cells."""
        if radius is None:
            # topology based
            raise NotImplementedError
        else:
            return cells_around(self.centers(), radius, frmt=frmt)

    def nodal_adjacency_matrix(self, *args, **kwargs):
        """Returns the nodal adjecency matrix."""
        topo = self.topology()
        return nodal_adjacency(topo, *args, **kwargs)

    def number_of_cells(self):
        """Returns the number of cells."""
        blocks = self.cellblocks(inclusive=True)
        return np.sum(list(map(lambda i: len(i.celldata), blocks)))

    def number_of_points(self):
        """Returns the number of points"""
        return len(self.root().pointdata)

    def cells_coords(self, *args, _topo=None, **kwargs):
        """Returns the coordiantes of the cells in extrensic format."""
        _topo = self.topology() if _topo is None else _topo
        return cells_coords(self.root().coords(), _topo)

    def center(self, target: FrameLike = None):
        """Returns the center of the pointcloud of the mesh."""
        if self.is_root():
            return self.points().center(target)
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            return pc.center(target)

    def centers(self, *args, target: FrameLike = None, **kwargs):
        """Returns the centers of the cells."""
        if self.is_root():
            coords = self.points().show(target)
        else:
            root = self.root()
            inds = np.unique(self.topology())
            pc = root.points()[inds]
            coords = pc.show(target)
        return cell_center_bulk(coords, self.topology(*args, **kwargs))

    def centralize(self, target: FrameLike = None):
        """Centralizes the coordinats of the pointcloud of the mesh."""
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
        """Returns the areas."""
        coords = self.root().coords()
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        blocks2d = filter(lambda b: b.celltype.NDIM < 3, blocks)
        amap = map(lambda b: b.celldata.areas(coords=coords), blocks2d)
        return np.concatenate(list(amap))

    def area(self, *args, **kwargs):
        """Returns the sum of areas in the model."""
        return np.sum(self.areas(*args, **kwargs))

    def volumes(self, *args, **kwargs):
        """Returns the volumes of the cells."""
        coords = self.root().coords()
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        vmap = map(lambda b: b.celldata.volumes(coords=coords), blocks)
        return np.concatenate(list(vmap))

    def volume(self, *args, **kwargs):
        """Returns the net volume of the mesh."""
        return np.sum(self.volumes(*args, **kwargs))

    def index_of_closest_point(self, target, *args, **kwargs):
        """Returns the index of the closest point to a target."""
        return index_of_closest_point(self.coords(), target)

    def index_of_closest_cell(self, target, *args, **kwargs):
        """Returns the index of the closest cell to a target."""
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

    def to_vtk(self, *args, deepcopy=True, fuse=True, deep=True,
               scalars=None, **kwargs):
        """
        Returns the mesh as a `vtk` oject, and optionally fetches
        data.
        
        """
        if not __hasvtk__:
            raise ImportError
        coords = self.root().coords()
        blocks = list(self.cellblocks(inclusive=True, deep=deep))
        if fuse:
            if len(blocks) == 1:
                topo = blocks[0].celldata.nodes.to_numpy().astype(np.int64)
                ugrid = mesh_to_vtk(*detach_mesh_bulk(coords, topo),
                                    blocks[0].celltype.vtkCellType, deepcopy)
                return ugrid
            mb = vtk.vtkMultiBlockDataSet()
            mb.SetNumberOfBlocks(len(blocks))
            for i, block in enumerate(blocks):
                topo = block.celldata.nodes.to_numpy().astype(np.int64)
                ugrid = mesh_to_vtk(*detach_mesh_bulk(coords, topo),
                                    block.celltype.vtkCellType, deepcopy)
                mb.SetBlock(i, ugrid)
            return mb
        else:
            needsdata = isinstance(scalars, str)
            res, plotdata = [], []
            for i, block in enumerate(blocks):
                if needsdata:
                    pdata = None
                    if self.pointdata is not None:
                        if scalars in self.pointdata.fields:
                            pdata = self.pointdata[scalars].to_numpy()
                    if pdata is None and scalars in self.celldata.fields:
                        pdata = self.celldata[scalars].to_numpy()
                    plotdata.append(pdata)
                # the next line handles regular topologies only
                topo = block.celldata.nodes.to_numpy().astype(np.int64)
                ugrid = mesh_to_vtk(*detach_mesh_bulk(coords, topo),
                                    block.celltype.vtkCellType, deepcopy)
                res.append(ugrid)

            if needsdata:
                return res, plotdata
            else:
                return res

    def to_pv(self, *args, fuse=True, deep=True, scalars=None, **kwargs):
        """
        Returns the mesh as a `pyVista` oject, optionally set up with data.
        
        """
        if not __haspyvista__:
            raise ImportError
        if isinstance(scalars, str) and not fuse:
            vtkobj, data = self.to_vtk(*args, fuse=fuse, deep=deep,
                                       scalars=scalars, **kwargs)
        else:
            vtkobj = self.to_vtk(*args, fuse=fuse, deep=deep, **kwargs)
            data = None
        if fuse:
            multiblock = pv.wrap(vtkobj)
            try:
                multiblock.wrap_nested()
            except AttributeError:
                pass
            return multiblock
        else:
            if data is None:
                return [pv.wrap(i) for i in vtkobj]
            else:
                res = []
                for ugrid, d in zip(vtkobj, data):
                    pvobj = pv.wrap(ugrid)
                    if isinstance(d, ndarray):
                        pvobj[scalars] = d
                    res.append(pvobj)
                return res

    def plot(self, *args, deepcopy=True, jupyter_backend='pythreejs',
             show_edges=True, notebook=False, theme='document',
             scalars=None, **kwargs):
        if not __haspyvista__:
            raise ImportError
        if theme is not None:
            pv.set_plot_theme(theme)
        poly = self.to_pv(deepcopy=deepcopy, sclars=scalars)
        if notebook:
            return poly.plot(*args, jupyter_backend=jupyter_backend,
                             show_edges=show_edges, notebook=notebook, **kwargs)
        else:
            poly.plot(*args, show_edges=show_edges, notebook=notebook,
                      **kwargs)

    def __join_parent__(self, parent: DeepDict, key: Hashable = None):
        super().__join_parent__(parent, key)
        if self.celldata is not None:
            GIDs = self.root().cim.generate_np(len(self.celldata))
            self.celldata['id'] = GIDs
        if self.pointdata is not None:
            GIDs = self.root().pim.generate_np(len(self.pointdata))
            self.pointdata['id'] = GIDs
        self.rewire(deep=True)

    def __repr__(self):
        return 'PolyData(%s)' % (dict.__repr__(self))


class IndexManager(object):
    """This object ought to guarantee, that every cell in a 
    model has a unique ID."""

    def __init__(self, start=0):
        self.queue = []
        self.next = start

    def generate_np(self, n=1):
        if n == 1:
            return self.generate(1)
        else:
            return np.array(self.generate(n))

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