"""
Tests for Balkwill symmetry analysis.
"""

import numpy as np

from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.balkwill import BalkwillTriangle



def test_balkwill_symmetry_difference():

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
        point=np.array([0.0, 80.0, 0.0]),
        reference_used="Dental midline"
    )


    balkwill = BalkwillTriangle(
        left_posterior=left_molar,
        right_posterior=right_molar,
        dental_midline=midline
    )
    assert np.isclose(
        balkwill.symmetry_difference,
        0.0
    )