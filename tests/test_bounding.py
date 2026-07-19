"""
OGDD Bounding Box Tests

Tests for axis aligned bounding boxes.
"""

import numpy as np
import pytest

from ogdd.geometry.bounding import BoundingBox



def test_create_bounding_box_from_points():
    """
    Test bounding box generation from points.
    """

    points = np.array(
        [
            [0, 0, 0],
            [10, 5, 2],
            [-3, 4, 8],
        ]
    )


    box = BoundingBox.from_points(
        points
    )


    assert np.allclose(
        box.min_point,
        [-3, 0, 0]
    )


    assert np.allclose(
        box.max_point,
        [10, 5, 8]
    )



def test_bounding_box_center():
    """
    Test center calculation.
    """

    box = BoundingBox(
        min_point=[0,0,0],
        max_point=[10,10,10]
    )


    center = box.center()


    assert np.allclose(
        center,
        [5,5,5]
    )



def test_bounding_box_size():
    """
    Test bounding dimensions.
    """

    box = BoundingBox(
        min_point=[0,0,0],
        max_point=[10,5,2]
    )


    size = box.size()


    assert np.allclose(
        size,
        [10,5,2]
    )



def test_bounding_box_diagonal():
    """
    Test diagonal length.
    """

    box = BoundingBox(
        min_point=[0,0,0],
        max_point=[3,4,0]
    )


    diagonal = box.diagonal()


    assert diagonal == 5



def test_invalid_point_cloud():
    """
    Test invalid point array.
    """

    with pytest.raises(ValueError):

        BoundingBox.from_points(
            [
                [1,2],
                [3,4]
            ]
        )



def test_invalid_min_point():
    """
    Test invalid bounding box dimensions.
    """

    with pytest.raises(ValueError):

        BoundingBox(
            min_point=[0,0],
            max_point=[1,1,1]
        )
