# -*- coding: utf-8 -*-
from awkward.highlevel import Array


class TopologyArray(Array):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)