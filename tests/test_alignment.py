"""
OGDD Alignment Tests

Tests for PCA based geometric alignment.
"""

import numpy as np
import pytest

from ogdd.algorithms.alignment import (
    Alignment
)



def test_alignment_centers_model():
    """
    Test that centroid becomes origin.
    """

    points = np.array(
        [
            [0,0,0],
            [10,0,0],
            [0,10,0],
            [0,0,10],
        ],
        dtype=float
    )


    result = Alignment.compute(
        points
    )


    centroid = np.mean(
        result.points,
        axis=0
    )


    assert np.allclose(
        centroid,
        [0,0,0],
        atol=1e-7
    )



def test_alignment_preserves_distances():
    """
    Rigid transformations must preserve distances.
    """

    points = np.array(
        [
            [0,0,0],
            [5,0,0],
            [0,5,0],
        ],
        dtype=float
    )


    result = Alignment.compute(
        points
    )


    original_distance = np.linalg.norm(
        points[1]-points[0]
    )


    aligned_distance = np.linalg.norm(
        result.points[1]-result.points[0]
    )


    assert np.isclose(
        original_distance,
        aligned_distance
    )



def test_alignment_returns_rotation():

    points = np.array(
        [
            [0,0,0],
            [10,0,0],
            [0,5,0],
            [0,0,2],
        ],
        dtype=float
    )


    result = Alignment.compute(
        points
    )


    assert result.rotation.shape == (
        3,
        3
    )



def test_alignment_requires_3d_points():

    with pytest.raises(ValueError):

        Alignment.compute(
            [
                [1,2],
                [3,4]
            ]
        )
