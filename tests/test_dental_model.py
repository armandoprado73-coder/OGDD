"""
Tests for DentalModel.
"""

import numpy as np

from ogdd.anatomy.dental_model import DentalModel
from ogdd.anatomy.landmark import Landmark
from ogdd.mesh import Mesh



def test_dental_model_initial_state():
    """
    A new model should not be ready for Balkwill triangle.
    """

    mesh = Mesh()

    model = DentalModel(mesh=mesh)

    assert model.landmark_count == 0
    assert model.is_balkwill_ready is False



def test_balkwill_ready_after_three_landmarks():
    """
    Model becomes ready when the three required landmarks exist.
    """

    mesh = Mesh()

    model = DentalModel(mesh=mesh)


    model.add_landmark(
        Landmark(
            name="RIGHT_SECOND_MOLAR",
            point=np.array([1.0, 0.0, 0.0]),
            reference_used="Right second molar cusp"
        )
    )


    model.add_landmark(
        Landmark(
            name="LEFT_SECOND_MOLAR",
            point=np.array([-1.0, 0.0, 0.0]),
            reference_used="Left second molar cusp"
        )
    )


    model.add_landmark(
        Landmark(
            name="DENTAL_MIDLINE",
            point=np.array([0.0, 1.0, 0.0]),
            reference_used="Dental midline"
        )
    )


    assert model.landmark_count == 3
    assert model.is_balkwill_ready is True