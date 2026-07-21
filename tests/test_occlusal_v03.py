"""
OGDD Occlusal Plane Estimator v0.3 Tests
Tests complete analysis pipeline.
"""

import numpy as np

from ogdd.algorithms.occlusal import (
    OcclusalPlaneEstimator
)

from ogdd.core.result import (
    AnalysisResult
)

from ogdd.geometry.plane import (
    Plane
)


def create_occlusal_surface():
    """Create a simple synthetic occlusal surface for testing."""

    return np.array(
        [
            [0, 0, 10],
            [10, 0, 10],
            [0, 10, 10],
            [10, 10, 10],
            [5, 5, 10],

            [0, 0, 1],
            [10, 10, 2],
            [5, 2, 3],
        ],
        dtype=float,
    )


def create_result(top_percentage=40):
    """Run the occlusal plane estimator on the synthetic dataset."""

    estimator = OcclusalPlaneEstimator(
        top_percentage=top_percentage
    )

    return estimator.compute(
        create_occlusal_surface()
    )


def test_returns_analysis_result():

    result = create_result()

    assert isinstance(
        result,
        AnalysisResult
    )


def test_result_contains_plane():

    result = create_result()

    assert isinstance(
        result.value,
        Plane
    )


def test_result_contains_confidence():

    result = create_result()

    assert result.confidence is not None


def test_result_metadata():

    result = create_result()

    assert "points_used" in result.metadata

    assert "top_percentage" in result.metadata


def test_pipeline_with_translated_model():

    points = (
        create_occlusal_surface()
        +
        np.array(
            [
                100,
                50,
                -20
            ]
        )
    )

    result = (
        OcclusalPlaneEstimator()
        .compute(points)
    )

    assert isinstance(
        result.value,
        Plane
    )
