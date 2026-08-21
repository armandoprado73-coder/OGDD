"""
Test creation of Bonwill triangle from landmarks.
"""

import numpy as np

from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.bonwill import BonwillTriangle



def test_bonwill_triangle_creation():
    """
    Test creation of Bonwill triangle from landmarks.
    """

    right_condyle = Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([50.0, 0.0, 0.0]),
        reference_used="Right Condyle"
    )


    left_condyle = Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-50.0, 0.0, 0.0]),
        reference_used="Left Condyle"
    )


    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 86.6, 0.0]),
        reference_used="Dental midline"
    )


    bonwill = BonwillTriangle(
        left_condyle=left_condyle,
        right_condyle=right_condyle,
        dental_midline=midline,
    )

    assert bonwill is not None


    assert np.array_equal(
        bonwill.triangle.a,
        right_condyle.point
    )


    assert np.array_equal(
        bonwill.triangle.b,
        left_condyle.point
    )


    assert np.array_equal(
        bonwill.triangle.c,
        midline.point
    )

def test_bonwill_plane():
    """
    Test geometric plane defined by
    the Bonwill triangle.
    """

    right_condyle = Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([50.0, 0.0, 0.0]),
        reference_used="Right condyle"
    )

    left_condyle = Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-50.0, 0.0, 0.0]),
        reference_used="Left condyle"
    )

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 86.6, 0.0]),
        reference_used="Dental midline"
    )

    bonwill = BonwillTriangle(
        left_condyle=left_condyle,
        right_condyle=right_condyle,
        dental_midline=midline
    )

    plane = bonwill.plane

    assert np.allclose(
        plane.normal,
        bonwill.triangle.normal
    )

    assert plane.distance(
        right_condyle.point
    ) == 0.0

    assert plane.distance(
        left_condyle.point
    ) == 0.0

    assert plane.distance(
        midline.point
    ) == 0.0