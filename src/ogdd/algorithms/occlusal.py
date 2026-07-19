"""
OGDD Occlusal Plane Estimator

Version 0.3

Complete analysis pipeline:

Alignment
Region extraction
Plane fitting
Confidence evaluation
Standard result output
"""

from __future__ import annotations


import numpy as np


from .alignment import Alignment
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
        Execute full pipeline.
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
            version="0.3",
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
            "alignment_origin",
            alignment.origin
        )


        return result
