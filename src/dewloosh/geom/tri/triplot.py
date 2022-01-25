# -*- coding: utf-8 -*-
from dewloosh.geom.tri.triang import triobj_to_mpl, get_triobj_data, \
    triangulate
from dewloosh.geom.tri.triutils import offset_tri
from dewloosh.geom.utils import cells_coords, explode_mesh_data_bulk
from dewloosh.core.tools import issequence
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Callable, Iterable
import numpy as np
from matplotlib.collections import PatchCollection
from functools import partial


__all__ = ['triplot']


def triplot(triobj, *args, hinton=False, data=None, title=None,
            label=None, fig=None, ax=None, axes=None, **kwargs):
    fig, axes = get_fig_axes(*args, data=data, ax=ax, axes=axes,
                             fig=fig, **kwargs)
    if isinstance(triobj, tuple):
        coords, topo = triobj
        triobj = triangulate(points=coords[:, :2], triangles=topo)[-1]
        coords, topo = None, None
    pdata = partial(triplot_data, triobj)
    pgeom = partial(triplot_geom, triobj)
    phint = partial(triplot_hinton, triobj)
    if data is not None:
        assert len(data.shape) <= 2, \
            "Data must be a 1 or 2 dimensional array."
        nD = 1 if len(data.shape) == 1 else data.shape[1]
        data = data.reshape((data.shape[0], nD))
        if not issequence(title):
            title = nD * (title, )
        if not issequence(label):
            label = nD * (label, )
        pfnc = phint if hinton else pdata
        axobj = [pfnc(ax, data[:, i], *args, fig=fig, title=title[i],
                      label=label[i], **kwargs)
                 for i, ax in enumerate(axes)]
        if nD == 1:
            data = data.reshape(data.shape[0])
    else:
        axobj = pgeom(axes[0], *args, fig=fig, title=title, **kwargs)
    return axobj


class TriPatchCollection(PatchCollection):

    def __init__(self, cellcoords, *args, **kwargs):
        pmap = map(lambda i: cellcoords[i], np.arange(len(cellcoords)))
        def fnc(points): return Polygon(points, closed=True)
        patches = list(map(fnc, pmap))
        super().__init__(patches, *args, **kwargs)


def triplot_hinton(triobj, ax, data, *args, lw=0.5, fcolor='b',
                   ecolor='k', title=None, suptitle=None, label=None,
                   **kwargs):
    axobj = None
    tri = triobj_to_mpl(triobj)
    points, triangles = get_triobj_data(tri, *args, trim2d=True, **kwargs)
    cellcoords = offset_tri(points, triangles, data)
    axobj = TriPatchCollection(cellcoords, fc=fcolor, ec=ecolor, lw=lw)
    ax.add_collection(axobj)
    _decorate_(ax=ax, points=points, title=title, suptitle=suptitle,
               label=label, **kwargs)
    return axobj


def triplot_geom(triobj, ax, *args, lw=0.5, marker='b-',
                 zorder=None, fcolor=None, ecolor='k',
                 fig=None, title=None, suptitle=None, label=None,
                 **kwargs):
    axobj = None
    tri = triobj_to_mpl(triobj)
    points, triangles = get_triobj_data(tri, *args, trim2d=True, **kwargs)

    if fcolor is None:
        if zorder is not None:
            axobj = ax.triplot(tri, marker, lw=lw, zorder=zorder, **kwargs)
        else:
            axobj = ax.triplot(tri, marker, lw=lw, **kwargs)
    else:
        cellcoords = cells_coords(points, triangles)
        axobj = TriPatchCollection(cellcoords, fc=fcolor, ec=ecolor, lw=lw)
        ax.add_collection(axobj)
    _decorate_(fig=fig, ax=ax, points=points, title=title,
               suptitle=suptitle, label=label, **kwargs)
    return axobj


def triplot_data(triobj, ax, data, *args, cmap='winter', fig=None,
                 ecolor='k', lw=0.1, title=None, suptitle=None,
                 label=None, **kwargs):

    axobj = None
    tri = triobj_to_mpl(triobj)
    points, triangles = get_triobj_data(tri, *args, trim2d=True, **kwargs)

    nData = len(data)
    if nData == len(triangles):
        nD = len(data.shape)
        if nD == 1:
            axobj = ax.tripcolor(tri, facecolors=data, cmap=cmap,
                                 edgecolors=ecolor, lw=lw)
        elif nD == 2 and data.shape[1] == 3:
            nT, nN = data.shape
            points, triangles, data = \
                explode_mesh_data_bulk(points, triangles, data)
            triobj = triangulate(points=points, triangles=triangles)[-1]
            tri = triobj_to_mpl(triobj)
            axobj = ax.tripcolor(tri, data, cmap=cmap,
                                 edgecolors=ecolor, lw=lw)
    elif nData == len(points):
        axobj = ax.tripcolor(tri, data, cmap=cmap,
                             edgecolors=ecolor, lw=lw)

    assert axobj is not None, "Failed to handle the provided data."
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(axobj, cax=cax)

    _decorate_(fig=fig, ax=ax, points=points, title=title,
               suptitle=suptitle, label=label, **kwargs)
    return axobj


def get_fig_axes(*args, data=None, fig=None, axes=None, shape=None,
                 horizontal=False, ax=None, **kwargs):
    if fig is not None:
        if axes is not None:
            return fig, axes
        elif ax is not None:
            return fig, (ax,)
    else:
        if data is not None:
            nD = 1 if len(data.shape) == 1 else data.shape[1]
            if nD == 1:
                try:
                    aspect = kwargs.get('aspect', 'equal')
                    args[0].set_aspect(aspect)
                    ax = args[0]
                except Exception:
                    fig, ax = plt.subplots()
                return fig, (ax,)
            if fig is None or axes is None:
                if shape is not None:
                    if isinstance(shape, int):
                        shape = (shape, 1) if horizontal else (1, shape)
                    assert nD == (shape[0] * shape[1]), \
                        "Mismatch in shape and data."
                else:
                    shape = (nD, 1) if horizontal else (1, nD)
                fig, axes = plt.subplots(*shape)
            if not isinstance(axes, Iterable):
                axes = (axes,)
            return fig, axes
        else:
            try:
                aspect = kwargs.get('aspect', 'equal')
                args[0].set_aspect(aspect)
                ax = args[0]
            except Exception:
                fig, ax = plt.subplots()
            return fig, (ax,)
    return None, None


def _decorate_(*args, fig=None, ax=None, aspect='equal', xlim=None,
               ylim=None, axis='on', offset=0.05, points=None,
               axfnc: Callable = None, title=None, suptitle=None,
               label=None, **kwargs):
    assert ax is not None, "A matplotlib Axes object must be provided as " \
        "keyword argument 'ax'!"
    if axfnc is not None:
        try:
            axfnc(ax)
        except Exception:
            raise RuntimeError('Something went wrong when calling axfnc.')
    if xlim is None:
        if points is not None:
            xlim = points[:, 0].min(), points[:, 0].max()
            if offset is not None:
                dx = np.abs(xlim[1] - xlim[0])
                xlim = xlim[0] - offset*dx, xlim[1] + offset*dx
    if ylim is None:
        if points is not None:
            ylim = points[:, 1].min(), points[:, 1].max()
            if offset is not None:
                dx = np.abs(ylim[1] - ylim[0])
                ylim = ylim[0] - offset*dx, ylim[1] + offset*dx
    ax.set_aspect(aspect)
    ax.axis(axis)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.set_xlabel(label)
    if fig is not None and suptitle is not None:
        fig.suptitle(suptitle)
    return ax


if __name__ == '__main__':
    pass
