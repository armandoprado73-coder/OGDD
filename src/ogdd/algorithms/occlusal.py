"""
OGDD Occlusal Plane Estimator

Version 0.2

Includes automatic alignment before
plane estimation.
"""

from __future__ import annotations


import numpy as np


from .alignment import Alignment
from .region import RegionSelector
from .plane_fitting import PlaneFitter

from ..geometry.plane import Plane



class OcclusalPlaneEstimator:
    """
    Automatic candidate occlusal plane estimator.
    """



    def __init__(
        self,
        top_percentage: float = 20.0
    ):

        self.top_percentage = (
            top_percentage
        )



    def compute(
        self,
        points: np.ndarray
    ) -> Plane:
        """
        Estimate occlusal plane.

        Steps:

        1. Align geometry.
        2. Extract upper region.
        3. Fit plane.
        """


        alignment = Alignment.compute(
            points
        )


        aligned_points = (
            alignment.points
        )


        selected_points = (
            RegionSelector.select_top_percentage(
                aligned_points,
                self.top_percentage
            )
        )


        if len(selected_points) < 3:

            raise ValueError(
                "Insufficient occlusal points."
            )


        plane = PlaneFitter.fit(
            selected_points
        )


        return plane
