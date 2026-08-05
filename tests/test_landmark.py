import numpy as np

from ogdd.anatomy import Landmark


def test_landmark_creation():

    point = np.array([1.0, 2.0, 3.0])

    landmark = Landmark(
        name="RIGHT_SECOND_MOLAR_DISTAL_CUSP",
        point=point,
        reference_used="SECOND_MOLAR"
    )

    assert landmark.name == "RIGHT_SECOND_MOLAR_DISTAL_CUSP"

    assert np.array_equal(landmark.point, point)

    assert landmark.reference_used == "SECOND_MOLAR"

    assert landmark.confidence == 1.0

    assert landmark.created_by == "operator"