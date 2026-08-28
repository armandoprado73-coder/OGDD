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
def test_rotation_about_z_axis():
    """
    Positive rotation follows the right-hand rule.
    """

    transform = Transform.rotation_about_axis(
        origin=[0, 0, 0],
        axis=[0, 0, 1],
        angle_degrees=90,
    )

    result = transform.apply(
        np.array(
            [
                [1, 0, 0],
            ]
        )
    )

    assert np.allclose(
        result,
        [
            [0, 1, 0],
        ],
        atol=1e-8,
    )


def test_rotation_about_arbitrary_origin():
    """
    Rotation axis may pass through any 3D point.
    """

    transform = Transform.rotation_about_axis(
        origin=[1, 2, 3],
        axis=[0, 0, 1],
        angle_degrees=90,
    )

    result = transform.apply(
        np.array(
            [
                [2, 2, 3],
            ]
        )
    )

    assert np.allclose(
        result,
        [
            [1, 3, 3],
        ],
        atol=1e-8,
    )


def test_rotation_axis_is_normalized():
    """
    Rotation must accept a non-unit axis.
    """

    unit_axis = Transform.rotation_about_axis(
        origin=[0, 0, 0],
        axis=[0, 0, 1],
        angle_degrees=45,
    )

    scaled_axis = Transform.rotation_about_axis(
        origin=[0, 0, 0],
        axis=[0, 0, 10],
        angle_degrees=45,
    )

    points = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
        ]
    )

    assert np.allclose(
        unit_axis.apply(points),
        scaled_axis.apply(points),
    )


def test_rotation_keeps_points_on_axis_fixed():
    """
    Points located on the rotation axis remain fixed.
    """

    transform = Transform.rotation_about_axis(
        origin=[1, 2, 3],
        axis=[0, 0, 1],
        angle_degrees=73,
    )

    points = np.array(
        [
            [1, 2, 3],
            [1, 2, 8],
            [1, 2, -4],
        ]
    )

    assert np.allclose(
        transform.apply(points),
        points,
    )


def test_rotation_preserves_distances():
    """
    Rotation must behave as a rigid transformation.
    """

    transform = Transform.rotation_about_axis(
        origin=[3, -2, 5],
        axis=[1, 2, 3],
        angle_degrees=37,
    )

    points = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [-2, 7, 1],
        ]
    )

    rotated = transform.apply(points)

    original_distance = np.linalg.norm(
        points[1] - points[0]
    )

    rotated_distance = np.linalg.norm(
        rotated[1] - rotated[0]
    )

    assert np.isclose(
        rotated_distance,
        original_distance,
    )


def test_zero_length_rotation_axis_is_rejected():
    """
    A zero-length vector cannot define an axis.
    """

    with pytest.raises(ValueError):

        Transform.rotation_about_axis(
            origin=[0, 0, 0],
            axis=[0, 0, 0],
            angle_degrees=30,
        )


def test_rotation_inverse_recovers_original_points():
    """
    Applying the inverse must recover the input points.
    """

    transform = Transform.rotation_about_axis(
        origin=[1, 2, 3],
        axis=[1, 1, 1],
        angle_degrees=28,
    )

    points = np.array(
        [
            [4, 5, 6],
            [-1, 3, 8],
        ]
    )

    rotated = transform.apply(points)

    recovered = transform.inverse().apply(
        rotated
    )

    assert np.allclose(
        recovered,
        points,
    )
def test_translation_does_not_change_vectors():
    """
    Direction vectors must not receive translation.
    """

    transform = Transform.translation(
        [10, 20, 30]
    )

    vectors = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    result = transform.apply_vectors(
        vectors
    )

    assert np.allclose(
        result,
        vectors,
    )


def test_rotation_applies_to_vectors():
    """
    Direction vectors receive rotation without
    translation around the rotation origin.
    """

    transform = Transform.rotation_about_axis(
        origin=[10, 20, 30],
        axis=[0, 0, 1],
        angle_degrees=90,
    )

    vectors = np.array(
        [
            [1, 0, 0],
        ]
    )

    result = transform.apply_vectors(
        vectors
    )

    assert np.allclose(
        result,
        [
            [0, 1, 0],
        ],
        atol=1e-8,
    )


def test_invalid_vector_array():
    """
    Direction vectors must have shape (N,3).
    """

    transform = Transform.identity()

    with pytest.raises(ValueError):

        transform.apply_vectors(
            [
                [1, 2],
            ]
        )