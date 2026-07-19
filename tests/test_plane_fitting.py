"""
OGDD Plane Fitting Tests

Tests for mathematical plane estimation.
"""

import numpy as np
import pytest

from ogdd.algorithms.plane_fitting import (
    PlaneFitter
)



def test_fit_xy_plane():
    """
    Test fitting plane Z=0.
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


    assert np.allclose(
        plane.point,
        [5,5,0]
    )


    # Normal direction can be +/-.
    assert np.isclose(
        abs(plane.normal[2]),
        1.0
    )



def test_fit_inclined_plane():
    """
    Test a tilted plane.

    Plane:

        z = x + y
    """

    points = np.array(
        [
            [0,0,0],
            [1,0,1],
            [0,1,1],
            [2,1,3],
            [1,2,3],
        ],
        dtype=float
    )


    plane = PlaneFitter.fit(
        points
    )


    distances = [
        plane.distance(p)
        for p in points
    ]


    assert np.allclose(
        distances,
        0,
        atol=1e-6
    )



def test_plane_normal_is_unit_vector():

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


    length = np.linalg.norm(
        plane.normal
    )


    assert np.isclose(
        length,
        1.0
    )



def test_invalid_plane_points():

    with pytest.raises(ValueError):

        PlaneFitter.fit(
            [
                [0,0],
                [1,1]
            ]
        )



def test_not_enough_points():

    with pytest.raises(ValueError):

        PlaneFitter.fit(
            [
                [0,0,0],
                [1,1,1]
            ]
        )
