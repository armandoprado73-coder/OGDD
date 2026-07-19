"""
OGDD Plane Fitting Algorithms

Computes mathematical planes from
3D point clouds.
"""

from __future__ import annotations

import numpy as np

from ..geometry.plane import Plane
from ..geometry.vectors import normalize



class PlaneFitter:
    """
    Fits planes from 3D point clouds.
    """



    @staticmethod
    def fit(
        points: np.ndarray
    ) -> Plane:
        """
        Calculate best fitting plane.

        Uses PCA.

        Parameters
        ----------
        points:
            Nx3 point cloud.

        Returns
        -------
        Plane
            Best fitting plane.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2:

            raise ValueError(
                "Points must be 2D."
            )


        if points.shape[1] != 3:

            raise ValueError(
                "Points must have shape (N,3)."
            )


        if len(points) < 3:

            raise ValueError(
                "At least three points required."
            )


        center = np.mean(
            points,
            axis=0
        )


        centered = (
            points - center
        )


        covariance = np.cov(
            centered,
            rowvar=False
        )


        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )


        normal = eigenvectors[
            :, 
            np.argmin(eigenvalues)
        ]


        normal = normalize(
            normal
        )


        return Plane(
            point=center,
            normal=normal
        )
