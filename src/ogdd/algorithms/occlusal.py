"""
OGDD Occlusal Plane Estimator

First experimental algorithm for
automatic occlusal plane estimation.
"""

from __future__ import annotations

import numpy as np

from .region import RegionSelector
from .plane_fitting import PlaneFitter
from ..geometry.plane import Plane



class OcclusalPlaneEstimator:
    """
    Estimates a candidate occlusal plane
    from dental surface points.
    """


    def __init__(
        self,
        top_percentage: float = 20.0
    ):
        """
        Parameters
        ----------
        top_percentage:
            Percentage of highest points
            used for plane calculation.
        """

        self.top_percentage = (
            top_percentage
        )



    def compute(
        self,
        points: np.ndarray
    ) -> Plane:
        """
        Calculate candidate occlusal plane.
        """


        selected_points = (
            RegionSelector.select_top_percentage(
                points,
                self.top_percentage
            )
        )


        if len(selected_points) < 3:

            raise ValueError(
                "Not enough points for plane estimation."
            )


        plane = PlaneFitter.fit(
            selected_points
        )


        return plane
