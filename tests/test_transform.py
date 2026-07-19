"""
OGDD Transform Tests

Tests for 3D homogeneous transformations.
"""

import numpy as np
import pytest

from ogdd.geometry.transform import Transform



def test_identity_transform():
    """
    Identity transform must not change points.
    """

    transform = Transform.identity()


    points = np.array(
        [
            [1,2,3],
            [4,5,6]
        ]
    )


    result = transform.apply(
        points
    )


    assert np.allclose(
        result,
        points
    )



def test_translation_transform():
    """
    Test translation operation.
    """

    transform = Transform.translation(
        [10,0,0]
    )


    points = np.array(
        [
            [0,0,0],
            [1,2,3]
        ]
    )


    result = transform.apply(
        points
    )


    expected = np.array(
        [
            [10,0,0],
            [11,2,3]
        ]
    )


    assert np.allclose(
        result,
        expected
    )



def test_scale_transform():
    """
    Test uniform scaling.
    """

    transform = Transform.scale(
        2
    )


    points = np.array(
        [
            [1,2,3]
        ]
    )


    result = transform.apply(
        points
    )


    expected = np.array(
        [
            [2,4,6]
        ]
    )


    assert np.allclose(
        result,
        expected
    )



def test_multiple_points_transform():
    """
    Test applying transform to point cloud.
    """

    transform = Transform.translation(
        [1,2,3]
    )


    points = np.zeros(
        (100,3)
    )


    result = transform.apply(
        points
    )


    assert result.shape == (
        100,
        3
    )


    assert np.allclose(
        result[0],
        [1,2,3]
    )



def test_invalid_matrix_size():
    """
    Test invalid transformation matrix.
    """

    with pytest.raises(ValueError):

        Transform(
            np.eye(3)
        )



def test_invalid_point_array():
    """
    Test invalid point dimensions.
    """

    transform = Transform.identity()


    with pytest.raises(ValueError):

        transform.apply(
            [
                [1,2]
            ]
        )
