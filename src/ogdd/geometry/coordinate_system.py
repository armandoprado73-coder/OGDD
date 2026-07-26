"""
OGDD Coordinate Systems

Defines explicit 3D coordinate reference systems.

A coordinate system is defined by:

    origin
    x_axis
    y_axis
    z_axis

The axes are stored as normalized vectors
forming a right-handed orthonormal basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vectors import normalize


@dataclass
class CoordinateSystem:
    """
    Explicit 3D coordinate reference system.
    """

    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


    def __post_init__(self) -> None:
        """
        Validate and normalize coordinate system axes.
        """

        self.origin = np.asarray(
            self.origin,
            dtype=float
        )

        self.x_axis = normalize(
            np.asarray(
                self.x_axis,
                dtype=float
            )
        )

        self.y_axis = normalize(
            np.asarray(
                self.y_axis,
                dtype=float
            )
        )

        self.z_axis = normalize(
            np.asarray(
                self.z_axis,
                dtype=float
            )
        )


        if self.origin.shape != (3,):
            raise ValueError(
                "Origin must have shape (3,)"
            )


        if self.x_axis.shape != (3,):
            raise ValueError(
                "X axis must have shape (3,)"
            )


        if self.y_axis.shape != (3,):
            raise ValueError(
                "Y axis must have shape (3,)"
            )


        if self.z_axis.shape != (3,):
            raise ValueError(
                "Z axis must have shape (3,)"
            )


    @classmethod
    def identity(
        cls
    ) -> "CoordinateSystem":
        """
        Create the standard world coordinate system.

        Origin:
            [0, 0, 0]

        X axis:
            [1, 0, 0]

        Y axis:
            [0, 1, 0]

        Z axis:
            [0, 0, 1]
        """

        return cls(
            origin=np.array(
                [0, 0, 0],
                dtype=float
            ),
            x_axis=np.array(
                [1, 0, 0],
                dtype=float
            ),
            y_axis=np.array(
                [0, 1, 0],
                dtype=float
            ),
            z_axis=np.array(
                [0, 0, 1],
                dtype=float
            )
        )


    @classmethod
    def from_three_points(
        cls,
        origin: np.ndarray,
        point_x: np.ndarray,
        point_y: np.ndarray
    ) -> "CoordinateSystem":
        """
        Create a coordinate system from three points.

        Parameters
        ----------
        origin:
            Origin of the coordinate system.

        point_x:
            Point defining the positive X direction.

        point_y:
            Point used to define the XY plane.

        Returns
        -------
        CoordinateSystem
            Right-handed orthonormal coordinate system.
        """

        origin = np.asarray(
            origin,
            dtype=float
        )

        point_x = np.asarray(
            point_x,
            dtype=float
        )

        point_y = np.asarray(
            point_y,
            dtype=float
        )


        x_axis = normalize(
            point_x - origin
        )


        vector_to_y = (
            point_y - origin
        )


        z_axis = normalize(
            np.cross(
                x_axis,
                vector_to_y
            )
        )


        y_axis = normalize(
            np.cross(
                z_axis,
                x_axis
            )
        )


        return cls(
            origin=origin,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis
        )


    def to_local(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Transform points from world coordinates
        into this coordinate system.

        Parameters
        ----------
        points:
            Nx3 point array.

        Returns
        -------
        numpy.ndarray
            Points expressed in local coordinates.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Points must have shape (N,3)"
            )


        relative = (
            points
            -
            self.origin
        )


        rotation = np.column_stack(
            [
                self.x_axis,
                self.y_axis,
                self.z_axis
            ]
        )


        return (
            relative
            @
            rotation
        )


    def to_world(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Transform points from local coordinates
        into world coordinates.

        Parameters
        ----------
        points:
            Nx3 point array.

        Returns
        -------
        numpy.ndarray
            Points expressed in world coordinates.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Points must have shape (N,3)"
            )


        rotation = np.column_stack(
            [
                self.x_axis,
                self.y_axis,
                self.z_axis
            ]
        )


        return (
            points
            @
            rotation.T
            +
            self.origin
        )