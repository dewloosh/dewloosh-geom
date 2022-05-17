# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.core.abc.wrap import Wrapper
from dewloosh.math.array import atleast2d

from .utils import avg_cell_data, distribute_nodal_data, homogenize_nodal_values


class CellData(Wrapper):

    def __init__(self, *args, pointdata=None, celldata=None,
                 wrap=None, **kwargs):
        if celldata is not None:
            wrap=celldata
        super().__init__(*args, wrap=wrap, **kwargs)
        self.pointdata = pointdata

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        try:
            return getattr(self._wrapped, attr)
        except AttributeError:
            try:
                if self.pointdata is not None:
                    if attr in self.pointdata.fields:
                        data = self.pointdata[attr].to_numpy()
                        topo = self._wrapped.nodes.to_numpy()
                        return avg_cell_data(data, topo)
            except:
                pass
            raise AttributeError("'{}' object has no attribute \
                called {}".format(self.__class__.__name__, attr))
        except Exception:
            raise AttributeError("'{}' object has no attribute \
                called {}".format(self.__class__.__name__, attr))

    def __len__(self):
        return len(self._wrapped)

    def set_nodal_distribution_factors(self, factors, key='ndf'):
        if len(factors) != len(self._wrapped):
            self._wrapped[key] = factors[self._wrapped.id]
        else:
            self._wrapped[key] = factors
            
    def pull(self, key: str=None, *args, ndfkey='ndf', store=False, 
             storekey=None, avg=False, data=None, **kwargs):
        storekey = key if storekey is None else storekey
        if key is not None:
            nodal_data = self.pointdata[key].to_numpy()
        else:
            assert isinstance(data, np.ndarray)
            nodal_data = data
        topo = self.nodes.to_numpy()
        ndf = self._wrapped[ndfkey].to_numpy()
        if len(nodal_data.shape) == 1:
            nodal_data = atleast2d(nodal_data, back=True)
        d = distribute_nodal_data(nodal_data, topo, ndf)
        # nE, nNE, nDATA
        if isinstance(avg, np.ndarray):
            assert len(avg.shape) == 1
            assert avg.shape[0] == d.shape[0]
            d = homogenize_nodal_values(d, avg)
            # nE, nDATA
        d = np.squeeze(d)
        if store:
            self._wrapped[key] = d
        return d

    def spull(self, *args, storekey=None, **kwargs):
        return self.pull(*args, store=True, storekey=storekey, **kwargs)
    
    def push(self, *args, **kwargs):
        raise NotImplementedError()
    
    def spush(self, *args, storekey=None, **kwargs):
        return self.push(*args, store=True, storekey=storekey, **kwargs)