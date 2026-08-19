"""
Tests for the anatomical coordinate system
defined from the Balkwill triangle.
"""

import numpy as np

from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.geometry.coordinate_system import CoordinateSystem


def test_balkwill_defines_anatomical_coordinate_system():
    """
    Test that the Balkwill triangle can define
    an anatomical coordinate system.
    """

    right_molar = Landmark(
        name="RIGHT_SECOND_MOLAR",
        point=np.array(
            [50.0, 0.0, 0.0]
        ),
        reference_used="Right second molar cusp"
    )

    left_molar = Landmark(
        name="LEFT_SECOND_MOLAR",
        point=np.array(
            [-50.0, 0.0, 0.0]
        ),
        reference_used="Left second molar cusp"
    )

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array(
            [0.0, 86.6, 0.0]
        ),
        reference_used="Dental midline"
    )

    balkwill = BalkwillTriangle(
        left_posterior=left_molar,
        right_posterior=right_molar,
        dental_midline=midline
    )

    coordinate_system = (
        balkwill.coordinate_system
    )

    assert np.allclose(
        coordinate_system.origin,
        midline.point
    )
    
    expected_x = (
        right_molar.point
        -
        left_molar.point
    )

    expected_x = expected_x / np.linalg.norm(
        expected_x
    )

    assert np.allclose(
        coordinate_system.x_axis,
        expected_x
    )

    vector_to_midline = (
        midline.point
        -
        left_molar.point
    )

    assert np.dot(
        coordinate_system.y_axis,
        vector_to_midline
    ) > 0

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

    calculated_z = np.cross(
        coordinate_system.x_axis,
        coordinate_system.y_axis
    )

    assert np.allclose(
        calculated_z,
        coordinate_system.z_axis
    )