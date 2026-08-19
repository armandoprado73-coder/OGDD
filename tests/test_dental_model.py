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

def test_coordinate_system_not_ready_without_landmarks():
    """
    Model should not be ready for coordinate system
    without the required anatomical landmarks.
    """

    mesh = Mesh()

    model = DentalModel(mesh=mesh)

    assert model.is_coordinate_system_ready is False



def test_coordinate_system_ready_after_three_landmarks():
    """
    Model becomes ready for coordinate system
    when the three required landmarks exist.
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

    assert model.is_coordinate_system_ready is True



def test_dental_model_builds_anatomical_coordinate_system():
    """
    DentalModel should construct the expected anatomical
    coordinate system from dental landmarks.
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

    coordinate_system = model.coordinate_system

    assert np.allclose(
        coordinate_system.origin,
        [0.0, 1.0, 0.0],
    )

    assert np.allclose(
        coordinate_system.x_axis,
        [1.0, 0.0, 0.0],
    )

    assert np.allclose(
        coordinate_system.y_axis,
        [0.0, 1.0, 0.0],
    )

    assert np.allclose(
        coordinate_system.z_axis,
        [0.0, 0.0, 1.0],
    )

def test_dental_model_coordinate_system_with_real_landmarks():
    """
    DentalModel should construct the anatomical coordinate system
    from landmarks selected on a real mandibular STL.
    """

    mesh = Mesh()

    model = DentalModel(mesh=mesh)

    model.add_landmark(
        Landmark(
            name="RIGHT_SECOND_MOLAR",
            point=np.array([
                -26.484565,
                13.030479,
                0.946036,
            ]),
            reference_used="Right second molar cusp"
        )
    )

    model.add_landmark(
        Landmark(
            name="LEFT_SECOND_MOLAR",
            point=np.array([
                29.139037,
                13.553044,
                0.271681,
            ]),
            reference_used="Left second molar cusp"
        )
    )

    model.add_landmark(
        Landmark(
            name="DENTAL_MIDLINE",
            point=np.array([
                -2.764405,
                -23.366814,
                3.742300,
            ]),
            reference_used="Dental midline"
        )
    )

    coordinate_system = model.coordinate_system

    assert np.allclose(
        coordinate_system.origin,
        [-2.764405, -23.366814, 3.742300],
    )

    assert np.allclose(
        coordinate_system.x_axis,
        [-0.9998824, -0.00939356, 0.01212211],
        atol=1e-6,
    )

    assert np.allclose(
        coordinate_system.y_axis,
        [0.01037699, -0.99642913, 0.08379319],
        atol=1e-6,
    )

    assert np.allclose(
        coordinate_system.z_axis,
        [0.01129171, 0.08390913, 0.99640943],
        atol=1e-6,
    )

def test_dental_model_and_balkwill_share_coordinate_system():
    """
    DentalModel and BalkwillTriangle should define
    the same anatomical coordinate system
    when they use the same landmarks.
    """

    mesh = Mesh()

    model = DentalModel(mesh=mesh)

    model.add_landmark(
        Landmark(
            name="RIGHT_SECOND_MOLAR",
            point=np.array([50.0, 0.0, 0.0]),
            reference_used="Right second molar cusp"
        )
    )

    model.add_landmark(
        Landmark(
            name="LEFT_SECOND_MOLAR",
            point=np.array([-50.0, 0.0, 0.0]),
            reference_used="Left second molar cusp"
        )
    )

    model.add_landmark(
        Landmark(
            name="DENTAL_MIDLINE",
            point=np.array([0.0, 86.6, 0.0]),
            reference_used="Dental midline"
        )
    )

    model_system = model.coordinate_system
    balkwill_system = model.balkwill_triangle.coordinate_system

    assert np.allclose(
        model_system.origin,
        balkwill_system.origin,
    )

    assert np.allclose(
        model_system.x_axis,
        balkwill_system.x_axis,
    )

    assert np.allclose(
        model_system.y_axis,
        balkwill_system.y_axis,
    )

    assert np.allclose(
        model_system.z_axis,
        balkwill_system.z_axis,
    )