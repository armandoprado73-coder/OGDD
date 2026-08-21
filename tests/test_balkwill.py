"""
Tests for Balkwill Triangle.
"""

import numpy as np

from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.balkwill import BalkwillTriangle



def test_balkwill_triangle_creation():
    """
    Test creation of Balkwill triangle from landmarks.
    """

    right_molar = Landmark(
        name="RIGHT_SECOND_MOLAR",
        point=np.array([50.0, 0.0, 0.0]),
        reference_used="Right second molar cusp"
    )


    left_molar = Landmark(
        name="LEFT_SECOND_MOLAR",
        point=np.array([-50.0, 0.0, 0.0]),
        reference_used="Left second molar cusp"
    )


    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 86.6, 0.0]),
        reference_used="Dental midline"
    )


    balkwill = BalkwillTriangle(
        left_posterior=left_molar,
        right_posterior=right_molar,
        dental_midline=midline
    )

    assert balkwill is not None


    assert np.array_equal(
        balkwill.triangle.a,
        right_molar.point
    )


    assert np.array_equal(
        balkwill.triangle.b,
        left_molar.point
    )


    assert np.array_equal(
        balkwill.triangle.c,
        midline.point
    )

def test_balkwill_plane():
    """
    Test geometric plane defined by
    the Balkwill triangle.
    """

    right_molar = Landmark(
        name="RIGHT_SECOND_MOLAR",
        point=np.array([50.0, 0.0, 0.0]),
        reference_used="Right second molar cusp"
    )

    left_molar = Landmark(
        name="LEFT_SECOND_MOLAR",
        point=np.array([-50.0, 0.0, 0.0]),
        reference_used="Left second molar cusp"
    )

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 86.6, 0.0]),
        reference_used="Dental midline"
    )

    balkwill = BalkwillTriangle(
        left_posterior=left_molar,
        right_posterior=right_molar,
        dental_midline=midline
    )

    plane = balkwill.plane

    assert np.allclose(
        plane.normal,
        balkwill.triangle.normal
    )

    assert plane.distance(
        right_molar.point
    ) == 0.0

    assert plane.distance(
        left_molar.point
    ) == 0.0

    assert plane.distance(
        midline.point
    ) == 0.0