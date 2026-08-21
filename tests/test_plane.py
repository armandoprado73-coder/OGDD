"""
OGDD Plane Geometry Tests

Tests for the 3D plane representation.
"""

import numpy as np
import pytest

from ogdd.geometry.plane import Plane
from ogdd.geometry.transform import Transform


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

def test_angle_between_parallel_planes():
    """
    Test angle between parallel planes.
    """

    plane_a = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )

    plane_b = Plane(
        point=[0, 0, 5],
        normal=[0, 0, 1]
    )

    angle = plane_a.angle_to(
        plane_b
    )

    assert angle == pytest.approx(0.0)



def test_angle_between_perpendicular_planes():
    """
    Test angle between perpendicular planes.
    """

    plane_a = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )

    plane_b = Plane(
        point=[0, 0, 0],
        normal=[1, 0, 0]
    )

    angle = plane_a.angle_to(
        plane_b
    )

    assert angle == pytest.approx(90.0)



def test_angle_between_planes_45_degrees():
    """
    Test a 45 degree angle between planes.
    """

    plane_a = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )

    plane_b = Plane(
        point=[0, 0, 0],
        normal=[0, 1, 1]
    )

    angle = plane_a.angle_to(
        plane_b
    )

    assert angle == pytest.approx(45.0)

def test_angle_between_opposite_normals():
    """
    Test that opposite normals represent
    the same plane orientation.
    """

    plane_a = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1]
    )

    plane_b = Plane(
        point=[0, 0, 0],
        normal=[0, 0, -1]
    )

    angle = plane_a.angle_to(
        plane_b
    )

    assert angle == pytest.approx(0.0)

def test_invalid_plane_point():
    """
    Test invalid point dimensions.
    """

    with pytest.raises(ValueError):

        Plane(
            point=[0, 0],
            normal=[0, 0, 1]
        )

def test_plane_transform():
    """
    Test transforming a plane by translation.
    """

    plane = Plane(
        point=[0, 0, 0],
        normal=[0, 0, 1],
    )

    transform = Transform.translation(
        [10, 20, 30]
    )

    transformed = plane.transform(
        transform
    )

    assert np.allclose(
        transformed.point,
        [10, 20, 30],
    )

    assert np.allclose(
        transformed.normal,
        [0, 0, 1],
    )
