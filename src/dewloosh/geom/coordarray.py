# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.core.tools import issequence
from dewloosh.math.linalg.vector import VectorBase, Vector
from dewloosh.math.linalg import ReferenceFrame as Frame
from dewloosh.math.array import i32array
from numba.core import types as nbtypes, cgutils
from numba.extending import typeof_impl, models, \
    make_attribute_wrapper, register_model, box, \
    unbox, NativeValue, type_callable, lower_builtin,\
    overload, overload_attribute
import operator


__all__ = ['CoordinateArray']
  

class CoordinateArray(Vector):
    
    _array_cls_ = VectorBase
    
    def __init__(self, *args, frame=None, inds=None, **kwargs):
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
        return CoordinateArray(arr, frame=self.frame, inds=inds)
    
    @property    
    def x(self):
        return self._array[:, 0] if len(self.shape) > 1 else self._array[0]
    
    @property    
    def y(self):
        return self._array[:, 1] if len(self.shape) > 1 else self._array[1]
    
    @property    
    def z(self):
        return self._array[:, 2] if len(self.shape) > 1 else self._array[2]
    
    def center(self):
        if self.dim == 1:
            return self
        foo = lambda i : np.mean(self._array[:, i])
        return np.array(list(map(foo, range(self.shape[1]))))
        
    def view(self, frame: Frame=None, *args, **kwargs):
        return self.array @ frame.dcm(source=self.frame).T
    

class CoordinateArrayType(nbtypes.Type):
    """Numba type."""

    def __init__(self, datatype, indstype=nbtypes.NoneType):
        self.data = datatype
        self.inds = indstype
        super(CoordinateArrayType, self).__init__(name='CoordinateArray')


make_attribute_wrapper(CoordinateArrayType, 'data', 'data')
make_attribute_wrapper(CoordinateArrayType, 'inds', 'inds')


@overload_attribute(CoordinateArrayType, 'x')
def attr_x(arr):
   def get(arr):
       return arr.data[:, 0]
   return get


@overload_attribute(CoordinateArrayType, 'y')
def attr_y(arr):
   def get(arr):
       return arr.data[:, 1]
   return get


@overload_attribute(CoordinateArrayType, 'z')
def attr_z(arr):
   def get(arr):
       return arr.data[:, 2]
   return get


@typeof_impl.register(CoordinateArray)
def type_of_impl(val, context):
    """`val` is the Python object being typed"""
    datatype = typeof_impl(val._array, context)
    indstype = typeof_impl(val.inds, context)
    return CoordinateArrayType(datatype, indstype)


@type_callable(CoordinateArray)
def type_of_callable(context):
    def typer(data, inds=None):
        datatype = typeof_impl(data, context)
        indstype = typeof_impl(inds, context) \
            if inds is not None else nbtypes.NoneType
        return CoordinateArrayType(datatype, indstype)
    return typer


@register_model(CoordinateArrayType)
class StructModel(models.StructModel):
    """Data model for nopython mode."""

    def __init__(self, dmm, fe_type):
        """
        fe_type is `CoordinateArrayType`
        """
        members = [
            ('data', fe_type.data),
            ('inds', fe_type.inds),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@overload(operator.getitem)
def overload_getitem(obj, idx):
    if isinstance(obj, CoordinateArrayType):

        def dummy_getitem_impl(obj, idx):
            return obj.data[idx]

        return dummy_getitem_impl
    

@lower_builtin(CoordinateArray, nbtypes.Array)
def lower_type(context, builder, sig, args):
    typ = sig.return_type
    data, inds = args
    obj = cgutils.create_struct_proxy(typ)(context, builder)
    obj.data = data
    obj.inds = inds
    return obj._getvalue()


@unbox(CoordinateArrayType)
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


@box(CoordinateArrayType)
def box_type(typ, val, c):
    """Convert a numba-native structure to a python object."""
    native_obj = cgutils.create_struct_proxy(
        typ)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(CoordinateArray))
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
    
    COORD = CoordinateArray([[0, 0, 0], [0, 0, 1.], 
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