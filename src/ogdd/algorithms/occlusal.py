"""
OGDD Occlusal Plane Estimator

Version 0.4

Complete analysis pipeline:

Region extraction
Plane fitting
Confidence evaluation
Standard result output

The estimator operates in the input
coordinate system.

The Z axis is considered the vertical
reference direction for region selection.
"""

from __future__ import annotations

import numpy as np

from .region import RegionSelector
from .plane_fitting import PlaneFitter
from .confidence import PlaneConfidence

from ..core.result import (
    AnalysisResult
)


class OcclusalPlaneEstimator:
    """
    Complete occlusal plane analysis.
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
    ) -> AnalysisResult:
        """
        Execute full occlusal plane analysis.

        The input coordinate system is preserved.

        The Z axis is used as the vertical
        reference direction for selecting
        the highest points.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2 or points.shape[1] != 3:

            raise ValueError(
                "Points must have shape (N,3)"
            )


        selected_points = (
            RegionSelector.select_top_percentage(
                points,
                self.top_percentage
            )
        )


        if len(selected_points) < 3:

            raise ValueError(
                "Insufficient points."
            )


        plane = PlaneFitter.fit(
            selected_points
        )


        confidence = PlaneConfidence.evaluate(
            plane,
            selected_points
        )


        result = AnalysisResult(
            value=plane,
            algorithm="OcclusalPlaneEstimator",
            version="0.4",
            confidence=confidence
        )


        result.add_metadata(
            "points_used",
            len(selected_points)
        )


        result.add_metadata(
            "top_percentage",
            self.top_percentage
        )


        result.add_metadata(
            "coordinate_system",
            "input"
        )


        return result