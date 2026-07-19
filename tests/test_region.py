"""
OGDD Region Selection Tests

Tests for geometric point filtering.
"""

import numpy as np
import pytest

from ogdd.algorithms.region import (
    RegionSelector
)



def test_select_by_height():
    """
    Test Z threshold filtering.
    """

    points = np.array(
        [
            [0,0,1],
            [1,0,5],
            [2,0,10],
            [3,0,2],
        ],
        dtype=float
    )


    result = RegionSelector.select_by_height(
        points,
        5
    )


    expected = np.array(
        [
            [1,0,5],
            [2,0,10],
        ]
    )


    assert np.allclose(
        result,
        expected
    )



def test_select_top_percentage():
    """
    Test highest percentage selection.
    """

    points = np.array(
        [
            [0,0,0],
            [1,0,1],
            [2,0,2],
            [3,0,3],
            [4,0,4],
        ],
        dtype=float
    )


    result = RegionSelector.select_top_percentage(
        points,
        40
    )


    assert len(result) == 2



def test_selection_preserves_coordinates():
    """
    Ensure points remain unchanged.
    """

    points = np.array(
        [
            [5,7,10],
            [1,2,0]
        ],
        dtype=float
    )


    result = RegionSelector.select_by_height(
        points,
        5
    )


    assert np.allclose(
        result[0],
        [5,7,10]
    )



def test_invalid_percentage_low():

    points = np.zeros(
        (10,3)
    )


    with pytest.raises(ValueError):

        RegionSelector.select_top_percentage(
            points,
            0
        )



def test_invalid_percentage_high():

    points = np.zeros(
        (10,3)
    )


    with pytest.raises(ValueError):

        RegionSelector.select_top_percentage(
            points,
            120
        )



def test_invalid_point_shape():

    with pytest.raises(ValueError):

        RegionSelector.select_by_height(
            [
                [1,2],
                [3,4]
            ],
            1
        )
