"""
OGDD Principal Component Analysis

Provides PCA-based orientation estimation
for 3D point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.vectors import normalize



@dataclass
class PCAResult:
    """
    Result of principal component analysis.
    """

    centroid: np.ndarray

    principal_axis: np.ndarray

    secondary_axis: np.ndarray

    normal_axis: np.ndarray

    eigenvalues: np.ndarray



class PrincipalComponentAnalysis:
    """
    PCA estimator for 3D point clouds.
    """


    @staticmethod
    def compute(
        points: np.ndarray
    ) -> PCAResult:
        """
        Calculate principal directions.

        Parameters
        ----------
        points:
            Nx3 point cloud.

        Returns
        -------
        PCAResult
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2:

            raise ValueError(
                "Points must be a 2D array."
            )


        if points.shape[1] != 3:

            raise ValueError(
                "Points must have shape (N,3)."
            )


        if len(points) < 3:

            raise ValueError(
                "At least 3 points are required."
            )


        centroid = np.mean(
            points,
            axis=0
        )


        centered = (
            points - centroid
        )


        covariance = np.cov(
            centered,
            rowvar=False
        )


        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )


        order = np.argsort(
            eigenvalues
        )[::-1]


        eigenvalues = eigenvalues[
            order
        ]

        eigenvectors = eigenvectors[
            :, order
        ]


        principal_axis = normalize(
            eigenvectors[:,0]
        )


        secondary_axis = normalize(
            eigenvectors[:,1]
        )


        normal_axis = normalize(
            eigenvectors[:,2]
        )


        return PCAResult(
            centroid=centroid,
            principal_axis=principal_axis,
            secondary_axis=secondary_axis,
            normal_axis=normal_axis,
            eigenvalues=eigenvalues
        )
