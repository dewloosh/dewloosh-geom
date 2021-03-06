# -*- coding: utf-8 -*-
import numpy as np

from dewloosh.math.linalg import ReferenceFrame as FrameLike
from dewloosh.math.array import isboolarray

from .space import CartesianFrame, PointCloud
from .akwrap import AkWrapper

def gen_frame(coords): return CartesianFrame(dim=coords.shape[1])


class PointData(AkWrapper):
    """This is a class"""

    _point_cls_ = PointCloud

    def __init__(self, *args, points=None, coords=None, wrap=None, fields=None,
                 frame: FrameLike = None, newaxis: int = 2, stateful=False, 
                 activity=None, **kwargs):
        fields = {} if fields is None else fields
        assert isinstance(fields, dict)
       
        # coordinate frame
        if not isinstance(frame, FrameLike):
            if coords is not None:
                frame = gen_frame(coords)
        self._frame = frame

        # set pointcloud
        point_cls = self.__class__._point_cls_
        X = None
        if len(args) > 0:
            if isinstance(args[0], np.ndarray):
                X = args[0]
        else:
            X = points if coords is None else coords
        assert isinstance(
            X, np.ndarray), 'Coordinates must be specified as a numpy array!'
        nP, nD = X.shape
        if nD == 2:
            inds = [0, 1, 2]
            inds.pop(newaxis)
            if isinstance(frame, FrameLike):
                if len(frame) == 3:
                    _c = np.zeros((nP, 3))
                    _c[:, inds] = X
                    X = _c
                    X = point_cls(X, frame=frame).show()
                elif len(frame) == 2:
                    X = point_cls(X, frame=frame).show()
        elif nD == 3:
            X = point_cls(X, frame=frame).show()
        fields['x'] = X
        
        if activity is None:
            activity = np.ones(nP, dtype=bool)
        else:
            assert isboolarray(activity) and len(activity.shape) == 1, \
                "'activity' must be a 1d boolean numpy array!"
        if activity is None and stateful:
            fields['active'] = np.ones(nP, dtype=bool)
        fields['active'] = activity
        
        if stateful:
            fields['active'] = np.ones(nP, dtype=bool)
            
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                if v.shape[0] == nP:
                    fields[k] = v
                
        super().__init__(*args, wrap=wrap, fields=fields, **kwargs)
        self._celldata = []
    
    @property
    def frame(self):
        return self._frame
