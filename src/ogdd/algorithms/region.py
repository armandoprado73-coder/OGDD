"""
OGDD Region Selection Algorithms

Provides basic geometric filtering
for extracting areas of interest.
"""

from __future__ import annotations

import numpy as np



class RegionSelector:
    """
    Geometric region extraction tools.
    """



    @staticmethod
    def select_by_height(
        points: np.ndarray,
        minimum_z: float
    ) -> np.ndarray:
        """
        Select points above a Z threshold.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2 or points.shape[1] != 3:

            raise ValueError(
                "Points must have shape (N,3)"
            )


        mask = (
            points[:,2]
            >=
            minimum_z
        )


        return points[
            mask
        ]



    @staticmethod
    def select_top_percentage(
        points: np.ndarray,
        percentage: float
    ) -> np.ndarray:
        """
        Select highest percentage of points.

        Example:

            20 means top 20%.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if percentage <= 0 or percentage > 100:

            raise ValueError(
                "Percentage must be between 0 and 100."
            )


        threshold = np.percentile(
            points[:,2],
            100 - percentage
        )


        return RegionSelector.select_by_height(
            points,
            threshold
        )
