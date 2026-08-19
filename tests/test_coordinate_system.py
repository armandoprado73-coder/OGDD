"""
OGDD Coordinate System Tests

Tests for explicit 3D coordinate reference systems.
"""

import numpy as np
import pytest

from ogdd.geometry.coordinate_system import CoordinateSystem


def test_coordinate_system_creation():
    """
    Test basic coordinate system creation.
    """

    coordinate_system = CoordinateSystem(
        origin=[0, 0, 0],
        x_axis=[1, 0, 0],
        y_axis=[0, 1, 0],
        z_axis=[0, 0, 1]
    )

    assert np.allclose(
        coordinate_system.origin,
        [0, 0, 0]
    )

    assert np.allclose(
        coordinate_system.x_axis,
        [1, 0, 0]
    )

    assert np.allclose(
        coordinate_system.y_axis,
        [0, 1, 0]
    )

    assert np.allclose(
        coordinate_system.z_axis,
        [0, 0, 1]
    )


def test_axes_are_normalized():
    """
    Test that axes are automatically normalized.
    """

    coordinate_system = CoordinateSystem(
        origin=[0, 0, 0],
        x_axis=[10, 0, 0],
        y_axis=[0, 5, 0],
        z_axis=[0, 0, 2]
    )
    
    assert np.allclose(
        coordinate_system.origin,
        [0, 0, 0]
    )

    assert np.allclose(
        coordinate_system.x_axis,
        [1, 0, 0]
    )

    assert np.allclose(
        coordinate_system.y_axis,
        [0, 1, 0]
    )

    assert np.allclose(
        coordinate_system.z_axis,
        [0, 0, 1]
    )


def test_from_three_points():
    """
    Test coordinate system created from three points.
    """

    origin = np.array(
        [0, 0, 0],
        dtype=float
    )

    point_x = np.array(
        [10, 0, 0],
        dtype=float
    )

    point_y = np.array(
        [0, 10, 0],
        dtype=float
    )

    coordinate_system = (
        CoordinateSystem.from_three_points(
            origin,
            point_x,
            point_y
        )
    )

    assert np.allclose(
        coordinate_system.origin,
        [0, 0, 0]
    )

    assert np.allclose(
        coordinate_system.x_axis,
        [1, 0, 0]
    )

    assert np.allclose(
        coordinate_system.y_axis,
        [0, 1, 0]
    )

    assert np.allclose(
        coordinate_system.z_axis,
        [0, 0, 1]
    )


def test_from_three_points_axes_are_orthogonal():
    """
    Test that generated axes are mutually orthogonal.
    """

    coordinate_system = (
        CoordinateSystem.from_three_points(
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0]
        )
    )

    assert np.isclose(
        np.dot(
            coordinate_system.x_axis,
            coordinate_system.y_axis
        ),
        0.0
    )

    assert np.isclose(
        np.dot(
            coordinate_system.x_axis,
            coordinate_system.z_axis
        ),
        0.0
    )

    assert np.isclose(
        np.dot(
            coordinate_system.y_axis,
            coordinate_system.z_axis
        ),
        0.0
    )


def test_from_three_points_axes_are_normalized():
    """
    Test that generated axes have unit length.
    """

    coordinate_system = (
        CoordinateSystem.from_three_points(
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0]
        )
    )

    assert np.isclose(
        np.linalg.norm(
            coordinate_system.x_axis
        ),
        1.0
    )

    assert np.isclose(
        np.linalg.norm(
            coordinate_system.y_axis
        ),
        1.0
    )

    assert np.isclose(
        np.linalg.norm(
            coordinate_system.z_axis
        ),
        1.0
    )


def test_from_three_points_right_handed_system():
    """
    Test right-handed coordinate system.

    X cross Y must equal Z.
    """

    coordinate_system = (
        CoordinateSystem.from_three_points(
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0]
        )
    )

    calculated_z = np.cross(
        coordinate_system.x_axis,
        coordinate_system.y_axis
    )

    assert np.allclose(
        calculated_z,
        coordinate_system.z_axis
    )


def test_invalid_origin_shape():
    """
    Test invalid origin dimensions.
    """

    with pytest.raises(ValueError):

        CoordinateSystem(
            origin=[0, 0],
            x_axis=[1, 0, 0],
            y_axis=[0, 1, 0],
            z_axis=[0, 0, 1]
        )


def test_invalid_x_axis_shape():
    """
    Test invalid X axis dimensions.
    """

    with pytest.raises(ValueError):

        CoordinateSystem(
            origin=[0, 0, 0],
            x_axis=[1, 0],
            y_axis=[0, 1, 0],
            z_axis=[0, 0, 1]
        )


def test_collinear_points_are_invalid():
    """
    Test that collinear points cannot define a coordinate system.
    """

    with pytest.raises(ValueError):

        CoordinateSystem.from_three_points(
            [0, 0, 0],
            [10, 0, 0],
            [20, 0, 0]
        )


def test_to_local_coordinates():
    """
    Test transformation from world to local coordinates.
    """

    coordinate_system = (
        CoordinateSystem.from_three_points(
            [10, 20, 30],
            [20, 20, 30],
            [10, 30, 30]
        )
    )

    points = np.array(
        [
            [10, 20, 30],
            [20, 20, 30],
            [10, 30, 30]
        ],
        dtype=float
    )

    local = coordinate_system.to_local(
        points
    )

    expected = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0]
        ],
        dtype=float
    )

    assert np.allclose(
        local,
        expected
    )


def test_to_world_coordinates():
    """
    Test transformation from local to world coordinates.
    """

    coordinate_system = (
        CoordinateSystem.from_three_points(
            [10, 20, 30],
            [20, 20, 30],
            [10, 30, 30]
        )
    )

    local_points = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0]
        ],
        dtype=float
    )

    world = coordinate_system.to_world(
        local_points
    )

    expected = np.array(
        [
            [10, 20, 30],
            [20, 20, 30],
            [10, 30, 30]
        ],
        dtype=float
    )

    assert np.allclose(
        world,
        expected
    )


def test_local_world_round_trip():
    """
    Test that local -> world -> local
    returns the original points.
    """

    coordinate_system = (
        CoordinateSystem.from_three_points(
            [10, 20, 30],
            [20, 20, 30],
            [10, 30, 30]
        )
    )

    original = np.array(
        [
            [0, 0, 0],
            [5, 3, 2],
            [10, 10, 5]
        ],
        dtype=float
    )

    world = coordinate_system.to_world(
        original
    )

    recovered = coordinate_system.to_local(
        world
    )

    assert np.allclose(
        recovered,
        original
    )


def test_identity_coordinate_system():
    """
    Test standard world coordinate system.
    """

    coordinate_system = (
        CoordinateSystem.identity()
    )

    assert np.allclose(
        coordinate_system.origin,
        [0, 0, 0]
    )

    assert np.allclose(
        coordinate_system.x_axis,
        [1, 0, 0]
    )

    assert np.allclose(
        coordinate_system.y_axis,
        [0, 1, 0]
    )

    assert np.allclose(
        coordinate_system.z_axis,
        [0, 0, 1]
    )

def test_dental_landmark_orientation():
    """
    Test coordinate system using dental landmarks.

    The left molar is the origin.
    The right molar defines the positive X direction.
    The dental midline defines the anterior direction.
    """

    left_molar = np.array(
        [0, 0, 0],
        dtype=float
    )

    right_molar = np.array(
        [100, 10, 0],
        dtype=float
    )

    dental_midline = np.array(
        [35, 80, 5],
        dtype=float
    )

    coordinate_system = (
        CoordinateSystem.from_three_points(
            origin=left_molar,
            point_x=right_molar,
            point_y=dental_midline
        )
    )

    # The left molar is the origin.
    assert np.allclose(
        coordinate_system.origin,
        left_molar
    )

    # X must point from the left molar toward
    # the right molar.
    expected_x = (
        right_molar - left_molar
    )

    expected_x = (
        expected_x
        /
        np.linalg.norm(expected_x)
    )

    assert np.allclose(
        coordinate_system.x_axis,
        expected_x
    )

    # All axes must be unit vectors.
    assert np.isclose(
        np.linalg.norm(
            coordinate_system.x_axis
        ),
        1.0
    )

    assert np.isclose(
        np.linalg.norm(
            coordinate_system.y_axis
        ),
        1.0
    )

    assert np.isclose(
        np.linalg.norm(
            coordinate_system.z_axis
        ),
        1.0
    )

    # The three axes must be mutually orthogonal.
    assert np.isclose(
        np.dot(
            coordinate_system.x_axis,
            coordinate_system.y_axis
        ),
        0.0
    )

    assert np.isclose(
        np.dot(
            coordinate_system.x_axis,
            coordinate_system.z_axis
        ),
        0.0
    )

    assert np.isclose(
        np.dot(
            coordinate_system.y_axis,
            coordinate_system.z_axis
        ),
        0.0
    )

    # The system must be right-handed.
    assert np.allclose(
        np.cross(
            coordinate_system.x_axis,
            coordinate_system.y_axis
        ),
        coordinate_system.z_axis
    )
def test_real_dental_landmarks():
    """
    Test coordinate system using real dental landmarks
    selected from the mandibular STL.
    """

    left_molar = np.array(
        [
            29.139037,
            13.553044,
            0.271681,
        ],
        dtype=float,
    )

    right_molar = np.array(
        [
            -26.484565,
            13.030479,
            0.946036,
        ],
        dtype=float,
    )

    dental_midline = np.array(
        [
            -2.764405,
            -23.366814,
            3.742300,
        ],
        dtype=float,
    )

    coordinate_system = (
        CoordinateSystem.from_dental_landmarks(
            right_molar=right_molar,
            left_molar=left_molar,
            dental_midline=dental_midline,
        )
    )

    # --------------------------------------------------------
    # El origen debe ser la línea media dental
    # --------------------------------------------------------

    assert np.allclose(
        coordinate_system.origin,
        dental_midline,
    )

    # --------------------------------------------------------
    # Los tres ejes deben ser unitarios
    # --------------------------------------------------------

    assert np.isclose(
        np.linalg.norm(
            coordinate_system.x_axis
        ),
        1.0,
    )

    assert np.isclose(
        np.linalg.norm(
            coordinate_system.y_axis
        ),
        1.0,
    )

    assert np.isclose(
        np.linalg.norm(
            coordinate_system.z_axis
        ),
        1.0,
    )

    # --------------------------------------------------------
    # Los tres ejes deben ser ortogonales
    # --------------------------------------------------------

    assert np.isclose(
        np.dot(
            coordinate_system.x_axis,
            coordinate_system.y_axis,
        ),
        0.0,
        atol=1e-10,
    )

    assert np.isclose(
        np.dot(
            coordinate_system.x_axis,
            coordinate_system.z_axis,
        ),
        0.0,
        atol=1e-10,
    )

    assert np.isclose(
        np.dot(
            coordinate_system.y_axis,
            coordinate_system.z_axis,
        ),
        0.0,
        atol=1e-10,
    )

    # --------------------------------------------------------
    # El sistema debe ser derecho
    #
    # X × Y = Z
    # --------------------------------------------------------

    assert np.allclose(
        np.cross(
            coordinate_system.x_axis,
            coordinate_system.y_axis,
        ),
        coordinate_system.z_axis,
    )