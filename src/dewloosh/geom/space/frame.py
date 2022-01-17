# -*- coding: utf-8 -*-
import numpy as np
from dewloosh.math.linalg.frame import ReferenceFrame
from dewloosh.math.linalg.vector import Vector
from typing import Union
from numpy import ndarray
from copy import deepcopy as dcopy


__all__ = ['CartesianFrame']


VectorLike = Union[Vector, ndarray]


class CartesianFrame(ReferenceFrame):

    """
    A field-specific reference frame to be used in problems related to 
    Euclidean geometry.

    It builds on top of `ReferenceFrame` from `dewloosh.math`, but adds
    the contept of 'origo', and some other applications related to the field.

    Examples
    --------
    Define a standard Cartesian frame and rotate it around axis 'Z'
    with an amount of 180 degrees:

    >>> A = CartesianFrame(dim=3)
    >>> B = A.orient_new('Body', [0, 0, np.pi], 'XYZ')

    To create a third frame that rotates from B the way B rotates from A, we
    can do

    >>> A = CartesianFrame(dim=3)
    >>> C = A.orient_new('Body', [0, 0, 2*np.pi], 'XYZ')

    or we can define it relative to B (this literally makes C to looke 
    in B like B looks in A)

    >>> C = CartesianFrame(B.axes, parent=B)

    """

    def __init__(self, axes=None, *args, dim=3, origo=None, **kwargs):
        axes = np.eye(dim) if axes is None else axes
        super().__init__(axes, *args, **kwargs)
        self._origo = origo

    def origo(self, target: ReferenceFrame = None) -> Vector:
        """
        Returns the origo of the current frame in ambient space
        or with respect to another frame.

        Parameters
        ----------
        target : ReferenceFrame, Optional
            A frame in which we want to get the origo of the current frame.
            A None value returns the origo of the current frame with respect
            to the root. Default is None.

        Returns
        -------
        Vector
            A vector defined in ambient space, the parent frame, 
            or the specified frame.

        Examples
        --------
        Define a standard Cartesian frame and rotate it around axis 'Z'
        with an amount of 180 degrees:

        >>> A = CartesianFrame()
        >>> B = A.orient_new('Body', [0, 0, 45*np.pi/180],  'XYZ')

        To get the origin of frame B:

        >>> B.origo()
        [0., 0., 0.]

        Move frame B (the motion is defined locally) and print the
        new point of origin with respect to A: 

        >>> B.move(Vector([1, 0, 0], frame=B))
        >>> B.origo(A)
        [0.7071, 0.7071, 0.]

        Of course, the point of origin of a frame with respect to itself
        must be a zero vector:

        >>> B.origo(B)
        [0., 0., 0.]

        Providing with no arguments returns the distance of origin with 
        respect to the root frame:

        >>> B.origo()  # same as B.origo(B.root())
        [0.7071, 0.7071, 0.]

        """
        if self.parent is None:
            if target is None:
                if isinstance(self._origo, ndarray):
                    return self._origo
                elif self._origo is None:
                    return np.zeros(len(self.axes))
                else:
                    raise NotImplementedError
            elif isinstance(target, ReferenceFrame):
                o_self = self._origo if self._origo is not None \
                    else np.zeros(len(self.axes))
                o_target = target.origo()
                return Vector(o_self - o_target).show(target)
        else:
            if target is None:
                if isinstance(self._origo, ndarray):
                    o = Vector(self._origo, frame=self.parent).show()
                    return self.parent.origo() + o
                else:
                    return self.parent.origo()
            elif isinstance(target, ReferenceFrame):
                o_self = self.origo()
                o_target = target.origo()
                return Vector(o_self - o_target).show(target)

    def move(self, d: VectorLike, frame: ReferenceFrame = None):
        """
        Moves the frame by shifting its origo.
        
        Parameters
        ----------
        d : VectorLike
            Vector or Array, the amount of the motion. 
            
        frame : ReferenceFrame, Optional
            A frame in which the input is defined if it is not a Vector.
            Default is None.

        Returns
        -------
        ReferenceFrame
            The object the function is called on.

        Examples
        --------
        
        >>> A = CartesianFrame()
        >>> v = Vector([1., 0., 0.], frame=A)
        >>> B = A.fork('Body', [0, 0, 45*np.pi/180], 'XYZ').move(v)
        
        Move the frame locally with the same amount
        
        >>> B.move(v.array, frame=B)
        
        """
        if not isinstance(d, Vector):
            d = Vector(d, frame=frame)
        if self._origo is None:
            self._origo = np.zeros(len(self.axes))
        if self.parent is not None:
            self._origo += d.show(self.parent)
        else:
            self._origo += d.show()
        return self

    def rotate(self, *args, **kwargs):
        """
        Applies a transformation to the frame in-place.
        """
        return self.orient(*args, **kwargs)

    def fork(self, *args, **kwargs):
        """
        Retures a new frame, as a child of the current one. 
        Optionally, a transformation can be provided and all the arguments 
        are passed to the `orient_new` function. Otherwise, a frame is 
        returned with identical orientation and position.
        """
        if (len(args) + len(kwargs)) == 0:
            dim = len(self.axes)
            return self.__class__(np.eye(dim), parent=self)
        else:
            return self.orient_new(*args, **kwargs)

    def copy(self, deepcopy=False):
        """
        Returns a shallow or deep copy of this object, depending of the
        argument `deepcopy` (default is False).
        """
        if deepcopy:
            return self.__class__(dcopy(self.axes), parent=self.parent)
        else:
            return self.__class__(self.axes, parent=self.parent)

    def join(self, parent: ReferenceFrame = None):
        """
        Sets this object as the 'child' to the provided frame.
        If there is no frame provided, the object is joined to the root.
        """
        parent = parent if parent is not None else self.root()
        self._array = self.show(parent)
        self.parent = parent
        return self


if __name__ == '__main__':

    A = CartesianFrame()
    B = A.orient_new('Body', [0, 0, 45*np.pi/180],  'XYZ')
    print(B.origo())
    B.move(Vector([1, 0, 0], frame=B))
    print(B.origo(A))
    print(B.origo(B))
    print(B.origo())
    B.move(-Vector([np.sqrt(2)/2, 0, 0]))
    print(B.origo(A))
    B.move(-Vector([0, np.sqrt(2)/2, 0]))
    print(B.origo(A))
    C = B.fork().rotate('Body', [0, 0, 45*np.pi/180],
                        'XYZ').move(-Vector([0, np.sqrt(2)/2, 0]))
    
    # 
    A = CartesianFrame()
    v = Vector([1., 0., 0.], frame=A)
    B = A.fork('Body', [0, 0, 45*np.pi/180], 'XYZ').move(v)
