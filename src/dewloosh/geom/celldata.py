# -*- coding: utf-8 -*-
from dewloosh.core.abc.wrap import Wrapper
from .utils import avg_cell_data


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
