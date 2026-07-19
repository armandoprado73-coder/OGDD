"""
OGDD Plane Geometry

Defines the mathematical representation
of a 3D plane.

A plane is defined by:

    point + normal vector

Equation:

    n · (x - p) = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vectors import normalize


@dataclass
class Plane:
    """
    Mathematical representation of a 3D plane.

    Parameters
    ----------
    point:
        Any point belonging to the plane.

    normal:
        Plane normal vector.

    Notes
    -----
    The normal vector is always stored normalized.
    """

    point: np.ndarray

    normal: np.ndarray


    def __post_init__(self) -> None:
        """
        Normalize and validate plane data.
        """

        self.point = np.asarray(
            self.point,
            dtype=float
        )

        self.normal = normalize(
            np.asarray(
                self.normal,
                dtype=float
            )
        )


        if self.point.shape != (3,):
            raise ValueError(
                "Plane point must have shape (3,)"
            )


    def signed_distance(
        self,
        point: np.ndarray
    ) -> float:
        """
        Calculate signed distance from point to plane.

        Positive and negative values indicate
        opposite sides of the plane.
        """

        point = np.asarray(
            point,
            dtype=float
        )

        return float(
            np.dot(
                point - self.point,
                self.normal
            )
        )


    def distance(
        self,
        point: np.ndarray
    ) -> float:
        """
        Calculate absolute distance from point
        to plane.
        """

        return abs(
            self.signed_distance(point)
        )


    def project(
        self,
        point: np.ndarray
    ) -> np.ndarray:
        """
        Project a point onto the plane.
        """

        point = np.asarray(
            point,
            dtype=float
        )

        distance = self.signed_distance(
            point
        )

        return point - distance * self.normal
