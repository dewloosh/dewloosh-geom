# -*- coding: utf-8 -*-
from dewloosh.core import Hierarchy
from dewloosh.math.array import isfloatarray, isintegerarray
from dewloosh.geom.utils import cell_coords_bulk, \
    detach_mesh_bulk as detach_mesh, cell_center_bulk
from dewloosh.geom.vtkutils import mesh_to_vtk
from dewloosh.geom.polygon import Triangle
from dewloosh.geom.Q4 import Q4 as Quadrilateral
from dewloosh.geom.polyhedron import HexaHedron, Wedge, TriquadraticHexaHedron
from dewloosh.geom.utils import index_of_closest_point, nodal_distribution_factors
from dewloosh.geom.topo import regularize, nodal_adjacency, cells_at_nodes
import awkward as ak
from typing import Iterable
from copy import copy
import numpy as np
import vtk
import pyvista as pv


class PolyData(Hierarchy):

    def __init__(self, *args, coords=None, topo=None, celltype=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pointdata = None
        self.celldata = None
        self.celltype = None
        self.cell_index_manager = None

        # set coordinates
        if coords is not None:
            assert self.is_root(), "Currently only the top-level structure \
                (root) can hold onto point data."
            assert isfloatarray(coords), "Coordinates must be provided \
                as a numpy array of floats."
            nP = coords.shape[0]
            activity = np.ones(nP, dtype=bool)
            self.pointdata = ak.zip(
                {'x': coords, 'active': activity}, depth_limit=1)
            self.cell_index_manager = IndexManager()

        # set topology
        if topo is not None:
            if celltype is None:
                if isinstance(topo, np.ndarray):
                    nNode = topo.shape[1]
                    if nNode == 8:
                        celltype = HexaHedron
                    elif nNode == 6:
                        celltype = Wedge
                    elif nNode == 4:
                        celltype = Quadrilateral
                    elif nNode == 3:
                        celltype = Triangle
                    elif nNode == 27:
                        celltype = TriquadraticHexaHedron
                else:
                    raise NotImplementedError
            assert celltype is not None

            root = self.root()
            assert isintegerarray(topo), "Topology must be provided as a \
                numpy array of integers."
            topo = topo.astype(np.int64)
            if root.cell_index_manager is not None:
                GIDs = np.array(
                    root.cell_index_manager.generate(topo.shape[0]))
                self.celldata = celltype(wrap=ak.zip({'nodes': topo, 'GID': GIDs},
                                                     depth_limit=1),
                                         pointdata=root.pointdata)
            else:
                self.celldata = celltype(wrap=ak.zip({'nodes': topo}, depth_limit=1),
                                         pointdata=root.pointdata)

            self.celltype = celltype

    def blocks(self, *args, inclusive=False, blocktype=None, deep=True, **kwargs):
        dtype = PolyData if blocktype is None else blocktype
        return self.containers(self, inclusive=inclusive, dtype=dtype, deep=deep)

    def cellblocks(self, *args, **kwargs):
        return filter(lambda i: i.celldata is not None, self.blocks(*args, **kwargs))

    def coords(self, *args, return_inds=False, **kwargs):
        if self.is_root():
            return self.pointdata.x.to_numpy()
        else:
            # returns a sorted array of unique indices
            inds = np.unique(self.topology())
            if return_inds:
                return self.root().coords()[inds, :], inds
            else:
                return self.root().coords()[inds, :]

    def topology(self, *args, return_inds=False, **kwargs):
        blocks = list(self.cellblocks(*args, inclusive=True, **kwargs))
        topo = list(map(lambda i: i.celldata.nodes.to_numpy(), blocks))
        shapes = np.array(list(map(lambda arr: arr.shape[1], topo)))
        if not np.all(shapes == shapes[0]):
            raise NotImplementedError
        else:
            topo = np.vstack(topo)
            if return_inds:
                inds = list(map(lambda i: i.celldata.GID.to_numpy(), blocks))
                return topo, np.concatenate(inds)
            else:
                return topo

    def cells_at_nodes(self, *args, **kwargs):
        topo = self.topology()
        return cells_at_nodes(topo, *args, **kwargs)

    def nodal_adjacency_matrix(self, *args, **kwargs):
        topo = self.topology()
        return nodal_adjacency(topo, *args, **kwargs)

    def number_of_cells(self):
        blocks = self.cellblocks(inclusive=True)
        return np.sum(list(map(lambda i: len(i.celldata), blocks)))

    def number_of_points(self):
        return len(self.root().pointdata)

    def cellcoords(self, *args, **kwargs):
        return cell_coords_bulk(self.root().coords(), self.topology(*args, **kwargs))

    def centers(self, *args, **kwargs):
        return cell_center_bulk(self.root().coords(), self.topology(*args, **kwargs))

    def areas(self, *args, **kwargs):
        coords = self.root().coords()
        blocks = self.cellblocks(*args, inclusive=True, **kwargs)
        blocks2d = filter(lambda b: b.celltype.NDIM == 2, blocks)
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
        return index_of_closest_point(self.root().pointdata.x, target)

    def set_nodal_distribution_factors(self, *args, assume_regular=False, key='ndf', **kwargs):
        volumes = self.volumes()
        topo, inds = self.topology(return_inds=True)
        argsort = np.argsort(inds)
        topo = topo[argsort]
        volumes = volumes[argsort]
        if not assume_regular:
            topo, _ = regularize(topo)
        factors = nodal_distribution_factors(topo, volumes)
        blocks = self.cellblocks(inclusive=True)
        def foo(b): return b.celldata.set_nodal_distribution_factors(
            factors, key=key)
        list(map(foo, blocks))

    def to_vtk(self, deepcopy=True, fuse=True):
        coords = self.coords()
        blocks = list(self.cellblocks(inclusive=True))
        if fuse:
            mb = vtk.vtkMultiBlockDataSet()
            mb.SetNumberOfBlocks(len(blocks))
            for i, block in enumerate(blocks):
                topo = block.celldata.nodes.to_numpy()
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
        if fuse:
            multiblock = pv.wrap(self.to_vtk(*args, fuse=True, **kwargs))
            multiblock.wrap_nested()
            return multiblock
        else:
            return [pv.wrap(i) for i in self.to_vtk(*args, fuse=False, **kwargs)]

    def plot(self, *args, deepcopy=True, jupyter_backend='pythreejs',
             show_edges=True, notebook=False, theme='document', **kwargs):
        if theme is not None:
            pv.set_plot_theme(theme)
        if notebook:
            pv.wrap(self.to_vtk(deepcopy)).plot(*args,
                                                jupyter_backend=jupyter_backend, show_edges=show_edges,
                                                notebook=notebook, **kwargs)
        else:
            pv.wrap(self.to_vtk(deepcopy)).plot(*args,
                                                show_edges=show_edges, notebook=notebook,
                                                **kwargs)

    def __join_parent__(self, parent: Hierarchy):
        self.parent = parent
        self._root = parent.root()
        if self._root.cell_index_manager is not None and self.celldata is not None:
            GIDs = np.array(
                self._root.cell_index_manager.generate(len(self.celldata)))
            self.celldata['GID'] = GIDs

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


if __name__ == '__main__':

    from dewloosh.geom.rgrid import rgrid

    size = Lx, Ly, Lz = 800, 600, 20
    shape = nx, ny, nz = 8, 6, 2
    vtkCellType = vtk.VTK_TRIQUADRATIC_HEXAHEDRON

    coords1, topo1 = rgrid(size=size, shape=shape, eshape='H27')
    coords2, topo2 = rgrid(size=size, shape=shape, eshape='H27',
                           origo=(0, 0, 100))
    coords = np.vstack([coords1, coords2])
    topo2 += coords1.shape[0]

    pd = PolyData(coords=coords)
    pd['group1']['mesh1'] = PolyData(topo=topo1,
                                     vtkCellType=vtkCellType)
    pd['group2', 'mesh2'] = PolyData(topo=topo2,
                                     vtkCellType=vtkCellType)

    pd.plot()
