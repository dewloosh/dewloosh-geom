# -*- coding: utf-8 -*-
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk as np2vtk, \
    numpy_to_vtkIdTypeArray as np2vtkId


def mesh_to_vtk(coords, topo, vtkCellType, deepcopy=True):
    # points
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(np2vtk(coords, deep=deepcopy))

    # cells
    topo_vtk = np.concatenate((np.ones((topo.shape[0], 1), dtype=topo.dtype) *
                               topo.shape[1], topo), axis=1).ravel()
    cells_vtk = vtk.vtkCellArray()
    cells_vtk.SetNumberOfCells(topo.shape[0])
    cells_vtk.SetCells(topo.shape[0], np2vtkId(topo_vtk, deep=deepcopy))

    # grid
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(vtkpoints)
    ugrid.SetCells(vtkCellType, cells_vtk)
    return ugrid
