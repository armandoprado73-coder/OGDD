"""
OGDD Confidence Metrics

Evaluation tools for geometric algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.plane import Plane



@dataclass
class ConfidenceResult:
    """
    Quality metrics for a geometric result.
    """

    mean_error: float

    max_error: float

    point_count: int

    score: float



class PlaneConfidence:
    """
    Evaluates plane fitting quality.
    """



    @staticmethod
    def evaluate(
        plane: Plane,
        points: np.ndarray
    ) -> ConfidenceResult:
        """
        Calculate plane confidence.
        """

        points = np.asarray(
            points,
            dtype=float
        )


        if points.ndim != 2:

            raise ValueError(
                "Points must be 2D."
            )


        distances = np.array(
            [
                abs(
                    plane.distance(point)
                )
                for point in points
            ]
        )


        mean_error = float(
            np.mean(distances)
        )


        max_error = float(
            np.max(distances)
        )


        point_count = len(points)


        # Simple first confidence model.
        #
        # Lower error = higher score.
        #

        score = 1 / (
            1 + mean_error
        )


        return ConfidenceResult(
            mean_error=mean_error,
            max_error=max_error,
            point_count=point_count,
            score=score
        )
