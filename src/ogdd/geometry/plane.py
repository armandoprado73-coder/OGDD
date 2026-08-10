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


@dataclass(frozen=True)
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

        point = np.asarray(
            self.point,
            dtype=float
        )

        normal = normalize(
            np.asarray(
                self.normal,
                dtype=float
            )
        )

        if point.shape != (3,):
            raise ValueError(
                "Plane point must have shape (3,)"
            )

        if normal.shape != (3,):
            raise ValueError(
                "Plane normal must have shape (3,)"
            )

        object.__setattr__(
            self,
            "point",
            point
        )

        object.__setattr__(
            self,
            "normal",
            normal
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

    
    def transform(
        self,
        transform
    ) -> "Plane":
        """
        Transform plane to another coordinate system.

        The plane point is transformed as a point.
        The plane normal is transformed as a vector.
        """

        transformed_point = transform.apply(
            self.point.reshape(1, 3)
        )[0]


        rotation = transform.matrix[:3, :3]


        transformed_normal = (
            rotation @ self.normal
        )


        return Plane(
            point=transformed_point,
            normal=transformed_normal
        )