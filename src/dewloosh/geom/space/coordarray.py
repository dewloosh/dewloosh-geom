# -*- coding: utf-8 -*-
from dewloosh.core.tools import issequence
from dewloosh.math.array import i32array, minmax
from dewloosh.math.linalg.vector import VectorBase, Vector
from dewloosh.math.linalg.frame import ReferenceFrame as FrameLike
from dewloosh.geom.space.frame import CartesianFrame
from numba.core import types as nbtypes, cgutils
from numba.extending import typeof_impl, models, \
    make_attribute_wrapper, register_model, box, \
    unbox, NativeValue, type_callable, lower_builtin,\
    overload, overload_attribute
import operator
from typing import Union
from numpy import ndarray
import numpy as np
from numba import njit, prange
from typing import Union
__cache = True


__all__ = ['PointCloud']


VectorLike = Union[Vector, ndarray]


@njit(nogil=True, parallel=True, cache=__cache)
def show_coords(dcm: np.ndarray, coords: np.ndarray):
    res = np.zeros_like(coords)
    for i in prange(coords.shape[0]):
        res[i] = dcm @ coords[i,:]
    return res


def dcoords(coords, v):
    res = np.zeros_like(coords)
    res[:, 0] = v[0]
    res[:, 1] = v[1]
    try:
        res[:, 2] = v[2]
    except IndexError:
        pass
    finally:
        return res


class PointCloud(Vector):
    
    """
    A class to support calculations related to points in Euclidean space.

    Examples
    --------
    Collect the points of a simple triangulation and get the center:

    >>> from dewloosh.geom.tri import triangulate
    >>> coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
    >>> coords = PointCloud(coords)
    >>> coords.center()
        array([400., 300.,   0.])
        
    Centralize and get center again:
    
    >>> coords.centralize()
    >>> coords.center()
        array([0., 0., 0.])
        
    Move the points in the global frame:
    
    >>> coords.move(np.array([1., 0., 0.]))    
    >>> coords.center()
        array([1., 0., 0.])
        
    Rotate the points with 90 degrees around global Z.
    Before we do so, let check the boundaries:
    
    >>> coords.x().min(), coords.x().max()
    (-400., 400.)
    
    >>> coords.y().min(), coords.y().max()
    (-300., 300.)
    
    Now centralize wrt. the global frame, rotate and check
    the boundaries again:
    
    >>> coords.rotate('Body', [0, 0, np.pi], 'XYZ')
    >>> coords.center()
        [1., 0., 0.]
        
    >>> coords.x().min(), coords.x().max()
    (-300., 300.)
    
    >>> coords.y().min(), coords.y().max()
    (-400., 400.)
    
    The object keeps track of indices after slicing, always
    referring to the top level array:
    
    >>> coords[10:50][[1, 2, 10]].inds
    array([11, 12, 20])
    
    """
    
    _array_cls_ = VectorBase
    _frame_cls_ = CartesianFrame
    
    def __init__(self, *args, frame=None, inds=None, **kwargs):
        if frame is None:
            if len(args) > 0 and isinstance(args[0], np.ndarray):
                frame = self._frame_cls_(dim=args[0].shape[1])
        super().__init__(*args, frame=frame, **kwargs)
        self.inds = inds if inds is None else i32array(inds)
                
    def __getitem__(self, key):
        inds = None
        key = (key,) if not isinstance(key, tuple) else key
        if isinstance(key[0], slice):
            slc = key[0]
            start, stop, step = slc.start, slc.stop, slc.step
            start = 0 if start == None else start
            step = 1 if step == None else step
            stop = self.shape[0] if stop == None else stop
            inds = list(range(start, stop, step))
        elif issequence(key[0]):
            inds = key[0]
        elif isinstance(key[0], int):
            inds = [key[0],]
        if inds is not None and self.inds is not None:
            inds = self.inds[inds]
        arr = self._array.__getitem__(key)
        return PointCloud(arr, frame=self.frame, inds=inds)
    
    @property    
    def frame(self):
        """
        Returns the frame the points are embedded in.
        """
        return self._frame
    
    @frame.setter    
    def frame(self, target: FrameLike):
        """
        Sets the frame. This changes the frame itself and results 
        in a transformation of coordinates.
        
        Parameters
        ----------
                   
        target : ReferenceFrame
            A target frame of reference.
            
        """
        if isinstance(target, FrameLike):
            self._array = self.show(target)
            self._frame = target
        else:
            raise TypeError('Value must be a {} instance'.format(FrameLike))
        
    def x(self, target: FrameLike=None):
        arr = self.show(target)
        return arr[:, 0] if len(self.shape) > 1 else arr[0]
    
    def y(self, target: FrameLike=None):
        arr = self.show(target)
        return arr[:, 1] if len(self.shape) > 1 else arr[1]
    
    def z(self, target: FrameLike=None):
        arr = self.show(target)
        return arr[:, 2] if len(self.shape) > 1 else arr[2]
    
    def bounds(self, target: FrameLike=None):
        arr = self.show(target)
        dim = arr.shape[1]
        res = np.zeros((dim, 2))
        res[0] = minmax(arr[:, 0])
        res[1] = minmax(arr[:, 1])
        if dim > 2:
            res[2] = minmax(arr[:, 2])
        return res
       
    def center(self, target: FrameLike = None):
        """
        Returns the center of the points in a specified frame, 
        or the root frame if there is no target provided.
        
        Parameters
        ----------
                   
        target : ReferenceFrame, Optional
            A frame of reference. Default is None.
            
        Returns
        -------
        VectorBase
            A numpy array.
        
        """
        arr = self.show(target)
        foo = lambda i : np.mean(arr[:, i])
        return np.array(list(map(foo, range(self.shape[1]))))
                   
    def show(self, target: FrameLike=None):
        """
        Returns the coordinates of the points in a specified frame, 
        or the root frame if there is no target provided.
        
        Parameters
        ----------
                   
        target : ReferenceFrame, Optional
            A frame of reference. Default is None.
            
        Returns
        -------
        VectorBase
            A numpy array.
        
        """
        x = super().show(target)
        buf = x + dcoords(x, self.frame.origo(target))
        return self._array_cls_(shape=buf.shape, buffer=buf, dtype=buf.dtype)
            
    def move(self, v : VectorLike, frame: FrameLike = None):
        """
        Moves the points wrt. to a specified frame, or the root 
        frame if there is no target provided. Returns the object
        for continuation.
        
        Parameters
        ----------
        
        v : Vector or Array, Optional
            An array of a vector. If provided as an array, the `frame`
            argument can be used to specify the parent frame in which the
            motion is tp be understood.
           
        frame : ReferenceFrame, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None.

        Returns
        -------
        ReferenceFrame
            The object the function is called on.
            
        Examples
        --------
        Collect the points of a simple triangulation and get the center:

        >>> from dewloosh.geom.tri import triangulate
        >>> coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
        >>> coords = PointCloud(coords)
        >>> coords.center()
            array([400., 300.,   0.])
            
        Move the points and get the center again:
        
        d = np.array(0., 1., 0.)
        >>> coords.move(d).move(d)
        >>> coords.center()
        array([400., 302.,   0.])
        
        """
        if not isinstance(v, Vector):
            v = Vector(v, frame=frame)
        arr = v.show(self.frame)
        self._array += dcoords(self._array, arr)
        return self
        
    def centralize(self, target: FrameLike = None):
        """
        Centralizes the coordinates wrt. to a specified frame,
        or the root frame if there is no target provided.
        
        Returns the object for continuation.
        
        Parameters
        ----------
                   
        target : ReferenceFrame, Optional
            A frame of reference. Default is None.
            
        Returns
        -------
        ReferenceFrame
            The object the function is called on.
                
        """
        return self.move(-self.center(target), target)
        
    def rotate(self, *args, **kwargs):
        """
        Applies a transformation to the coordinates in-place. All arguments
        are passed to `ReferenceFrame.orient_new`, see its docs to know more.
        
        Returns the object for continuation.
        
        Examples
        --------
        To apply a 90 degree rotation about the Z axis:
        
        >>> coords.rotate('Body', [0, 0, np.pi/2], 'XYZ')
        
        """
        if isinstance(args[0], FrameLike):
            self._array = (self.frame.dcm(target=args[0]) @ self.array.T).T
            return self
        else:
            target = self.frame.orient_new(*args, **kwargs)
            return self.rotate(target)            


class PointCloudType(nbtypes.Type):
    """Numba type."""

    def __init__(self, datatype, indstype=nbtypes.NoneType):
        self.data = datatype
        self.inds = indstype
        super(PointCloudType, self).__init__(name='PointCloud')


make_attribute_wrapper(PointCloudType, 'data', 'data')
make_attribute_wrapper(PointCloudType, 'inds', 'inds')


@overload_attribute(PointCloudType, 'x')
def attr_x(arr):
   def get(arr):
       return arr.data[:, 0]
   return get


@overload_attribute(PointCloudType, 'y')
def attr_y(arr):
   def get(arr):
       return arr.data[:, 1]
   return get


@overload_attribute(PointCloudType, 'z')
def attr_z(arr):
   def get(arr):
       return arr.data[:, 2]
   return get


@typeof_impl.register(PointCloud)
def type_of_impl(val, context):
    """`val` is the Python object being typed"""
    datatype = typeof_impl(val._array, context)
    indstype = typeof_impl(val.inds, context)
    return PointCloudType(datatype, indstype)


@type_callable(PointCloud)
def type_of_callable(context):
    def typer(data, inds=None):
        datatype = typeof_impl(data, context)
        indstype = typeof_impl(inds, context) \
            if inds is not None else nbtypes.NoneType
        return PointCloudType(datatype, indstype)
    return typer


@register_model(PointCloudType)
class StructModel(models.StructModel):
    """Data model for nopython mode."""

    def __init__(self, dmm, fe_type):
        """
        fe_type is `PointCloudType`
        """
        members = [
            ('data', fe_type.data),
            ('inds', fe_type.inds),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@overload(operator.getitem)
def overload_getitem(obj, idx):
    if isinstance(obj, PointCloudType):

        def dummy_getitem_impl(obj, idx):
            return obj.data[idx]

        return dummy_getitem_impl
    

@lower_builtin(PointCloud, nbtypes.Array)
def lower_type(context, builder, sig, args):
    typ = sig.return_type
    data, inds = args
    obj = cgutils.create_struct_proxy(typ)(context, builder)
    obj.data = data
    obj.inds = inds
    return obj._getvalue()


@unbox(PointCloudType)
def unbox_type(typ, obj, c):
    """Convert a python object to a numba-native structure."""
    data_obj = c.pyapi.object_getattr_string(obj, "_array")
    inds_obj = c.pyapi.object_getattr_string(obj, "inds")
    native_obj = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    native_obj.data = c.unbox(typ.data, data_obj).value
    native_obj.inds = c.unbox(typ.inds, inds_obj).value
    c.pyapi.decref(data_obj)
    c.pyapi.decref(inds_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(native_obj._getvalue(), is_error=is_error)


@box(PointCloudType)
def box_type(typ, val, c):
    """Convert a numba-native structure to a python object."""
    native_obj = cgutils.create_struct_proxy(
        typ)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(PointCloud))
    data_obj = c.box(typ.data, native_obj.data)
    inds_obj = c.box(typ.inds, native_obj.inds)
    python_obj = c.pyapi.call_function_objargs(class_obj, (data_obj, inds_obj))
    c.pyapi.decref(data_obj)
    c.pyapi.decref(inds_obj)
    return python_obj
    

if __name__ == '__main__':
    from numba import njit

    @njit
    def foo(arr):
        return arr.inds
    
    COORD = PointCloud([[0, 0, 0], [0, 0, 1.], 
                             [0, 0, 0]],
                            inds=np.array([0, 1, 2, 3]))
    print(COORD[:, :].inds)
    arr1 = COORD[1:]
    print(arr1.inds)
    arr2 = arr1[1:]
    print(arr2.inds)
    print(COORD.to_numpy())
    
    print(foo(COORD))
    print(type(COORD @ np.eye(3)))
    
    from dewloosh.geom.tri import triangulate
    coords, topo, _ = triangulate(size=(800, 600), shape=(10, 10))
    coords = PointCloud(coords)
    print(coords.center())
    coords.centralize()
    print(coords.center())
    
    frameA = CartesianFrame(axes=np.eye(3), origo=np.array([-500., 0., 0.]))
    frameB = CartesianFrame(axes=np.eye(3), origo=np.array([+500., 0., 0.]))
        
    print("\norigos")
    print(frameA.origo())
    print(frameB.origo())
    
    print("\ncenters")
    print("center in global : {}".format(coords.center()))
    print("center in global : {}".format(coords.center(frameA)))
    print("center in global : {}".format(coords.center(frameB)))
    a = 1