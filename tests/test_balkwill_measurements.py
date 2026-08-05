"""
Tests for Balkwill clinical measurements.
"""

import numpy as np

from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.balkwill import BalkwillTriangle



def test_balkwill_measurements():

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
    
    assert np.isclose(
        balkwill.intermolar_width,
        100.0
    )


    assert np.isclose(
        balkwill.right_side,
        100.0,
        atol=0.01
    )


    assert np.isclose(
        balkwill.left_side,
        100.0,
        atol=0.01
    )