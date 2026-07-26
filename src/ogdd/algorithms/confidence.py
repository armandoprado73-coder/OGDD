"""
OGDD Plane Confidence Evaluation

Evaluates the quality of a fitted plane
using distances from points to the plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ogdd.geometry.plane import Plane


@dataclass
class ConfidenceResult:
    """
    Result of plane confidence evaluation.

    Parameters
    ----------
    mean_error:
        Mean absolute distance of the points
        from the fitted plane.

    point_count:
        Number of points used for evaluation.

    score:
        Confidence score between 0 and 1.
    """

    mean_error: float
    point_count: int
    score: float


class PlaneConfidence:
    """
    Evaluate the quality of a fitted plane.
    """

    @staticmethod
    def evaluate(
        plane: Plane,
        points: np.ndarray
    ) -> ConfidenceResult:
        """
        Evaluate plane quality from point-to-plane distances.
        """

        points = np.asarray(
            points,
            dtype=float
        )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Points must have shape (N, 3)"
            )

        if len(points) == 0:
            raise ValueError(
                "At least one point is required"
            )

        distances = np.array(
            [
                plane.distance(point)
                for point in points
            ],
            dtype=float
        )

        mean_error = float(
            np.mean(distances)
        )

        point_count = int(
            len(points)
        )

        score = float(
            1.0 / (1.0 + mean_error)
        )

        return ConfidenceResult(
            mean_error=mean_error,
            point_count=point_count,
            score=score
        )