"""
OGDD Bounding Box Geometry

Axis aligned bounding box utilities
for 3D mesh analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np



@dataclass
class BoundingBox:
    """
    Axis aligned bounding box.

    Parameters
    ----------
    min_point:
        Minimum coordinates.

    max_point:
        Maximum coordinates.
    """

    min_point: np.ndarray

    max_point: np.ndarray


    def __post_init__(self) -> None:

        self.min_point = np.asarray(
            self.min_point,
            dtype=float
        )

        self.max_point = np.asarray(
            self.max_point,
            dtype=float
        )


        if self.min_point.shape != (3,):

            raise ValueError(
                "min_point must have shape (3,)"
            )


        if self.max_point.shape != (3,):

            raise ValueError(
                "max_point must have shape (3,)"
            )



    @classmethod
    def from_points(
        cls,
        points: np.ndarray
    ) -> "BoundingBox":
        """
        Create bounding box from point cloud.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2 or points.shape[1] != 3:

            raise ValueError(
                "Points must have shape (N,3)"
            )


        return cls(
            min_point=np.min(
                points,
                axis=0
            ),

            max_point=np.max(
                points,
                axis=0
            )
        )



    def center(self) -> np.ndarray:
        """
        Return geometric center.
        """

        return (
            self.min_point +
            self.max_point
        ) / 2



    def size(self) -> np.ndarray:
        """
        Return dimensions along XYZ.
        """

        return (
            self.max_point -
            self.min_point
        )



    def diagonal(self) -> float:
        """
        Return diagonal length.
        """

        return float(
            np.linalg.norm(
                self.size()
            )
        )
