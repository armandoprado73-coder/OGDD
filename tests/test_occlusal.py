"""
OGDD Occlusal Plane Tests

Tests for first occlusal plane estimator.
"""

import numpy as np
import pytest

from ogdd.algorithms.occlusal import (
    OcclusalPlaneEstimator
)



def test_occlusal_plane_horizontal_surface():
    """
    Test flat dental-like surface.

    All points lie on Z=10.
    """

    points = np.array(
        [
            [0,0,10],
            [10,0,10],
            [0,10,10],
            [10,10,10],
            [5,5,10],

            # lower points
            [0,0,0],
            [5,5,2],
            [10,10,3],
        ],
        dtype=float
    )


    estimator = OcclusalPlaneEstimator(
        top_percentage=40
    )


    plane = estimator.compute(
        points
    )


    assert np.isclose(
        abs(plane.normal[2]),
        1.0
    )



def test_occlusal_plane_point():

    points = np.array(
        [
            [0,0,5],
            [10,0,5],
            [0,10,5],
            [10,10,5],
        ],
        dtype=float
    )


    estimator = OcclusalPlaneEstimator(
        top_percentage=100
    )


    plane = estimator.compute(
        points
    )


    assert np.allclose(
        plane.point,
        [5,5,5]
    )



def test_occlusal_plane_requires_points():

    points = np.array(
        [
            [0,0,1],
            [1,1,2]
        ],
        dtype=float
    )


    estimator = OcclusalPlaneEstimator()


    with pytest.raises(ValueError):

        estimator.compute(
            points
        )



def test_custom_percentage():

    estimator = OcclusalPlaneEstimator(
        top_percentage=10
    )


    assert (
        estimator.top_percentage
        ==
        10
    )
