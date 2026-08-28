"""
OGDD Geometry Transformations

Provides basic transformations for 3D geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np



@dataclass
class Transform:
    """
    General 3D transformation.

    Parameters
    ----------
    matrix:
        4x4 homogeneous transformation matrix.
    """

    matrix: np.ndarray


    def __post_init__(self):
        self.matrix = np.asarray(
            self.matrix,
            dtype=float
        )


        if self.matrix.shape != (4,4):
            raise ValueError(
                "Transformation matrix must be 4x4."
            )



    @classmethod
    def identity(
        cls
    ) -> "Transform":
        """
        Create identity transformation.
        """

        return cls(
            np.eye(4)
        )



    @classmethod
    def translation(
        cls,
        vector
    ) -> "Transform":
        """
        Create translation transform.
        """

        matrix = np.eye(4)


        matrix[:3, 3] = np.asarray(
            vector,
            dtype=float
        )


        return cls(matrix)


    @classmethod
    def scale(
        cls,
        factor
    ) -> "Transform":
        """
        Create uniform scale transform.
        """

        matrix = np.eye(4)


        matrix[0,0] = factor
        matrix[1,1] = factor
        matrix[2,2] = factor


        return cls(matrix)

    @classmethod
    def rotation_about_axis(
        cls,
        origin,
        axis,
        angle_degrees,
    ) -> "Transform":
        """
        Create a rotation around an arbitrary 3D axis.

        The axis passes through ``origin`` and follows
        the supplied direction. Positive angles follow
        the right-hand rule.
        """

        origin = np.asarray(
            origin,
            dtype=float,
        )

        axis = np.asarray(
            axis,
            dtype=float,
        )

        if origin.shape != (3,):
            raise ValueError(
                "Origin must be a 3D vector."
            )

        if axis.shape != (3,):
            raise ValueError(
                "Axis must be a 3D vector."
            )

        axis_length = np.linalg.norm(
            axis
        )

        if np.isclose(axis_length, 0.0):
            raise ValueError(
                "Rotation axis cannot have zero length."
            )

        axis = axis / axis_length

        angle_radians = np.deg2rad(
            float(angle_degrees)
        )

        cosine = np.cos(angle_radians)
        sine = np.sin(angle_radians)

        x, y, z = axis

        cross_matrix = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ]
        )

        rotation = (
            np.eye(3) * cosine
            + np.outer(axis, axis)
            * (1.0 - cosine)
            + cross_matrix * sine
        )

        matrix = np.eye(4)

        matrix[:3, :3] = rotation

        matrix[:3, 3] = (
            origin
            - rotation @ origin
        )

        return cls(matrix)

    def apply(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Apply transformation to points.

        Parameters
        ----------
        points:
            Nx3 point array.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Points must have shape (N,3)"
            )


        ones = np.ones(
            (len(points),1)
        )


        homogeneous = np.hstack(
            [
                points,
                ones
            ]
        )


        transformed = (
            self.matrix @ homogeneous.T
        ).T


        return transformed[:, :3]

    def apply_vectors(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the rotational part of the transformation
        to 3D direction vectors.

        Direction vectors do not receive translation.
        """

        vectors = np.asarray(
            vectors,
            dtype=float,
        )

        if (
            vectors.ndim != 2
            or vectors.shape[1] != 3
        ):
            raise ValueError(
                "Vectors must have shape (N,3)"
            )

        return (
            self.matrix[:3, :3]
            @ vectors.T
        ).T

    def inverse(
        self
    ) -> "Transform":
        """
        Return the inverse transformation.
        """

        return Transform(
            np.linalg.inv(
                self.matrix
            )
        )
