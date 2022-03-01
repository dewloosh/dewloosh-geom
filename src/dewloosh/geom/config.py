# -*- coding: utf-8 -*-
import configparser


try:
    import vtk
    __hasvtk__ = True
except Exception:
    __hasvtk__ = False
try:
    import pyvista as pv
    __haspyvista__ = True
except Exception:
    __haspyvista__ = False


def set_config_file(filepath):
    try:
        import vtk
        __hasvtk__ = True
    except Exception:
        __hasvtk__ = False
    try:
        import pyvista as pv
        __haspyvista__ = True
    except Exception:
        __haspyvista__ = False
    
    config = configparser.ConfigParser()
    config['geom'] = {}
    config['geom']['vtk'] = __hasvtk__
    config['geom']['vista'] = __haspyvista__
    with open('config.ini', 'w') as configfile:
        config.write(configfile)