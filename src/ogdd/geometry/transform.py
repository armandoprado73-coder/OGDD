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


        matrix[:3,3] = np.asarray(
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
