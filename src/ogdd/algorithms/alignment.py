"""
OGDD Alignment Algorithms

Provides automatic orientation
of 3D dental geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pca import (
    PrincipalComponentAnalysis
)

from ..geometry.transform import (
    Transform
)



@dataclass
class AlignmentResult:
    """
    Result of geometric alignment.
    """

    points: np.ndarray

    rotation: np.ndarray

    origin: np.ndarray



class Alignment:
    """
    PCA based alignment.
    """



    @staticmethod
    def compute(
        points: np.ndarray
    ) -> AlignmentResult:
        """
        Align point cloud to PCA frame.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        pca = (
            PrincipalComponentAnalysis.compute(
                points
            )
        )


        origin = pca.centroid


        rotation = np.column_stack(
            [
                pca.principal_axis,
                pca.secondary_axis,
                pca.normal_axis
            ]
        )


        transform = Transform(
            np.block(
                [
                    [
                        rotation.T,
                        -rotation.T @ origin.reshape(3,1)
                    ],
                    [
                        np.zeros((1,3)),
                        np.ones((1,1))
                    ]
                ]
            )
        )


        aligned_points = transform.apply(
            points
        )


        return AlignmentResult(
            points=aligned_points,
            rotation=rotation,
            origin=origin
        )
