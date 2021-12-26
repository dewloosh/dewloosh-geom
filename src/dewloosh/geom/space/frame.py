# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.math.linalg.frame import ReferenceFrame


class CartesianFrame(ReferenceFrame):
    
    def __init__(self, axes=None, *args, dim=3, **kwargs):
        axes = np.eye(dim) if axes is None else axes
        super().__init__(axes, *args, **kwargs)
