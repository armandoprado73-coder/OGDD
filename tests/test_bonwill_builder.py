"""
Tests for virtual Bonwill construction.
"""

import numpy as np

from ogdd.anatomy.landmark import Landmark
from ogdd.articulator.bonwill_builder import BonwillBuilder
from ogdd.articulator.configuration import (
    ArticulatorConfiguration,
)
from ogdd.geometry.coordinate_system import CoordinateSystem
from ogdd.geometry.plane import Plane

def test_build_default_bonwill_triangle():

    coordinate_system = CoordinateSystem.identity()

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 0.0, 0.0]),
        reference_used="Dental midline",
    )

    configuration = ArticulatorConfiguration()

    bonwill = BonwillBuilder.build(
        coordinate_system=coordinate_system,
        dental_midline=midline,
        configuration=configuration,
    )

    assert np.isclose(
        bonwill.condylar_width,
        110.0,
    )

    assert np.isclose(
        bonwill.right_side,
        110.0,
    )

    assert np.isclose(
        bonwill.left_side,
        110.0,
    )


def test_default_condyles_are_symmetric():

    coordinate_system = CoordinateSystem.identity()

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 0.0, 0.0]),
        reference_used="Dental midline",
    )

    configuration = ArticulatorConfiguration()

    bonwill = BonwillBuilder.build(
        coordinate_system=coordinate_system,
        dental_midline=midline,
        configuration=configuration,
    )

    right = bonwill.right_condyle.point
    left = bonwill.left_condyle.point

    assert np.isclose(right[0], 55.0)
    assert np.isclose(left[0], -55.0)

    assert np.isclose(
        right[1],
        left[1],
    )

    assert np.isclose(
        right[2],
        left[2],
    )


def test_condyles_are_posterior_and_superior():

    coordinate_system = CoordinateSystem.identity()

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 0.0, 0.0]),
        reference_used="Dental midline",
    )

    configuration = ArticulatorConfiguration()

    bonwill = BonwillBuilder.build(
        coordinate_system=coordinate_system,
        dental_midline=midline,
        configuration=configuration,
    )

    right = bonwill.right_condyle.point
    left = bonwill.left_condyle.point

    assert right[1] < 0.0
    assert left[1] < 0.0

    assert right[2] > 0.0
    assert left[2] > 0.0


def test_bonwill_is_symmetric():

    coordinate_system = CoordinateSystem.identity()

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 0.0, 0.0]),
        reference_used="Dental midline",
    )

    configuration = ArticulatorConfiguration()

    bonwill = BonwillBuilder.build(
        coordinate_system=coordinate_system,
        dental_midline=midline,
        configuration=configuration,
    )

    assert np.isclose(
        bonwill.symmetry_difference,
        0.0,
    )

def test_bonwill_plane_matches_configured_balkwill_angle():

    coordinate_system = CoordinateSystem.identity()

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 0.0, 0.0]),
        reference_used="Dental midline",
    )

    configuration = ArticulatorConfiguration()

    bonwill = BonwillBuilder.build(
        coordinate_system=coordinate_system,
        dental_midline=midline,
        configuration=configuration,
    )

    reference_plane = Plane(
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    measured_angle = reference_plane.angle_to(
        bonwill.plane
    )

    assert np.isclose(
        measured_angle,
        configuration.balkwill_angle_degrees,
    )

def test_bonwill_angle_is_preserved_in_rotated_coordinate_system():

    coordinate_system = CoordinateSystem(
        origin=np.array([10.0, 20.0, 30.0]),
        x_axis=np.array([1.0, 0.0, 0.0]),
        y_axis=np.array([0.0, 0.0, 1.0]),
        z_axis=np.array([0.0, -1.0, 0.0]),
    )

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=coordinate_system.origin,
        reference_used="Dental midline",
    )

    configuration = ArticulatorConfiguration()

    bonwill = BonwillBuilder.build(
        coordinate_system=coordinate_system,
        dental_midline=midline,
        configuration=configuration,
    )

    reference_plane = Plane(
        point=coordinate_system.origin,
        normal=coordinate_system.z_axis,
    )

    measured_angle = reference_plane.angle_to(
        bonwill.plane
    )

    assert np.isclose(
        measured_angle,
        configuration.balkwill_angle_degrees,
    )