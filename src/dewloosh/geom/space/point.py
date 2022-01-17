# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.math.linalg import Vector
from dewloosh.geom.space import StandardFrame


class Point(Vector):
    
    _frame_cls_ = StandardFrame
    
    def __init__(self, *args, frame=None, **kwargs):
        if frame is None:
            if len(args) > 0 and isinstance(args[0], np.ndarray):
                frame = self._frame_cls_(dim=args[0].shape[1])
        super().__init__(*args, frame=frame, **kwargs)