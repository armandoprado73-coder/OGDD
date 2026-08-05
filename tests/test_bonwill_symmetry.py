"""
Tests for Bonwill symmetry analysis.
"""

import numpy as np

from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.bonwill import BonwillTriangle



def test_bonwill_symmetry_difference():

    right = Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([50.0, 0.0, 0.0]),
        reference_used="Right condyle"
    )


    left = Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-50.0, 0.0, 0.0]),
        reference_used="Left condyle"
    )


    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 80.0, 0.0]),
        reference_used="Dental midline"
    )


    bonwill = BonwillTriangle(
        left_condyle=left,
        right_condyle=right,
        dental_midline=midline,
    )


    assert np.isclose(
        bonwill.symmetry_difference,
        0.0
    )