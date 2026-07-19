"""
OGDD Plane Geometry Tests

Tests for the 3D plane representation.
"""

import numpy as np
import pytest

from ogdd.geometry.plane import Plane



def test_plane_creation():
    """
    Test basic plane creation.
    """

    plane = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )


    assert np.allclose(
        plane.point,
        [0, 0, 0]
    )


    assert np.allclose(
        plane.normal,
        [0, 0, 1]
    )



def test_plane_normal_is_normalized():
    """
    Test that plane normal is automatically
    converted to unit length.
    """

    plane = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 10]
    )


    assert np.allclose(
        plane.normal,
        [0, 0, 1]
    )



def test_signed_distance_positive_side():
    """
    Test distance above the XY plane.
    """

    plane = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )


    distance = plane.signed_distance(
        [0, 0, 5]
    )


    assert distance == 5



def test_signed_distance_negative_side():
    """
    Test distance below the plane.
    """

    plane = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )


    distance = plane.signed_distance(
        [0, 0, -3]
    )


    assert distance == -3



def test_absolute_distance():
    """
    Test absolute point-plane distance.
    """

    plane = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )


    distance = plane.distance(
        [0, 0, -8]
    )


    assert distance == 8



def test_point_projection():
    """
    Test orthogonal projection onto plane.
    """

    plane = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )


    projected = plane.project(
        [2, 3, 5]
    )


    assert np.allclose(
        projected,
        [2, 3, 0]
    )



def test_invalid_plane_point():
    """
    Test invalid point dimensions.
    """

    with pytest.raises(ValueError):

        Plane(
            point=[0, 0],
            normal=[0, 0, 1]
        )
