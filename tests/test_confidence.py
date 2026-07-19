"""
OGDD Confidence Tests

Tests for plane quality evaluation.
"""

import numpy as np
import pytest

from ogdd.algorithms.plane_fitting import (
    PlaneFitter
)

from ogdd.algorithms.confidence import (
    PlaneConfidence
)



def test_perfect_plane_confidence():
    """
    Perfect plane should have zero error.
    """

    points = np.array(
        [
            [0,0,0],
            [10,0,0],
            [0,10,0],
            [10,10,0],
        ],
        dtype=float
    )


    plane = PlaneFitter.fit(
        points
    )


    confidence = PlaneConfidence.evaluate(
        plane,
        points
    )


    assert np.isclose(
        confidence.mean_error,
        0
    )


    assert confidence.point_count == 4



def test_noisy_plane_has_error():
    """
    A noisy surface should produce error.
    """

    points = np.array(
        [
            [0,0,0],
            [10,0,0.2],
            [0,10,-0.1],
            [10,10,0.3],
        ],
        dtype=float
    )


    plane = PlaneFitter.fit(
        points
    )


    confidence = PlaneConfidence.evaluate(
        plane,
        points
    )


    assert (
        confidence.mean_error
        >
        0
    )



def test_confidence_score_range():

    points = np.array(
        [
            [0,0,0],
            [5,0,0],
            [0,5,0],
        ],
        dtype=float
    )


    plane = PlaneFitter.fit(
        points
    )


    confidence = PlaneConfidence.evaluate(
        plane,
        points
    )


    assert (
        0
        <=
        confidence.score
        <=
        1
    )



def test_invalid_points():

    points = [
        [1,2],
        [3,4]
    ]


    plane_points = np.array(
        [
            [0,0,0],
            [1,0,0],
            [0,1,0]
        ],
        dtype=float
    )


    plane = PlaneFitter.fit(
        plane_points
    )


    with pytest.raises(ValueError):

        PlaneConfidence.evaluate(
            plane,
            points
        )
