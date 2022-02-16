# -*- coding: utf-8 -*-
import numpy as np
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk as np2vtk, \
        numpy_to_vtkIdTypeArray as np2vtkId
    __hasvtk__ = True
except Exception:
    __hasvtk__ = False


def mesh_to_vtkdata(coords, topo, deepcopy=True):
    if not __hasvtk__:
        raise ImportError
    # points
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(np2vtk(coords, deep=deepcopy))

    # cells
    topo_vtk = np.concatenate((np.ones((topo.shape[0], 1), dtype=topo.dtype) *
                               topo.shape[1], topo), axis=1).ravel()
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetNumberOfCells(topo.shape[0])
    vtkcells.SetCells(topo.shape[0], np2vtkId(topo_vtk, deep=deepcopy))
    
    return vtkpoints, vtkcells


def mesh_to_UnstructuredGrid(coords, topo, vtkCellType, deepcopy=True):
    vtkpoints, vtkcells = mesh_to_vtkdata(coords, topo, deepcopy)
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(vtkpoints)
    ugrid.SetCells(vtkCellType, vtkcells)
    return ugrid


def mesh_to_PolyData(coords, topo, deepcopy=True):
    vtkpoints, vtkcells = mesh_to_vtkdata(coords, topo, deepcopy)
    vtkPolyData = vtk.vtkPolyData()
    vtkPolyData.SetPoints(vtkpoints)
    vtkPolyData.SetPolys(vtkcells)
    vtkPolyData.Modified()
    return vtkPolyData
