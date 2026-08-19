"""
Test transformation of Balkwill landmarks
into the anatomical coordinate system.
"""

import numpy as np

from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.mesh import Mesh

def test_balkwill_landmarks_transform_to_local_coordinates():
    """
    Test that Balkwill landmarks are correctly positioned
    in the OGDD anatomical coordinate system.

    The dental midline is the origin.
    Positive X points to the patient's right.
    Positive Y points anteriorly.
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

    points = np.array(
        [
            left_molar.point,
            right_molar.point,
            midline.point
        ],
        dtype=float
    )

    local = balkwill.coordinate_system.to_local(
        points
    )

    # Left molar lies posterior and to the patient's left.
    assert np.allclose(
        local[0],
        [-50.0, -86.6, 0.0]
    )

    # Right molar lies posterior and to the patient's right.
    assert np.allclose(
        local[1],
        [50.0, -86.6, 0.0]
    )

    # Dental midline is the anatomical origin.
    assert np.allclose(
        local[2],
        [0.0, 0.0, 0.0]
    )

def test_mesh_vertices_transform_to_balkwill_local_coordinates():
    """
    Test that mesh vertices can be transformed
    into the Balkwill anatomical coordinate system.
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

    mesh = Mesh(
        vertices=np.array(
            [
                left_molar.point,
                right_molar.point,
                midline.point,
                [0.0, 0.0, 0.0]
            ],
            dtype=float
        )
    )

    local_vertices = (
        balkwill.coordinate_system.to_local(
            mesh.vertices
        )
    )

    # Left molar.
    assert np.allclose(
        local_vertices[0],
        [-50.0, -86.6, 0.0]
    )

    # Right molar.
    assert np.allclose(
        local_vertices[1],
        [50.0, -86.6, 0.0]
    )

    # Dental midline is the origin.
    assert np.allclose(
        local_vertices[2],
        [0.0, 0.0, 0.0]
    )

    # Original world origin expressed in anatomical coordinates.
    assert np.allclose(
        local_vertices[3],
        [0.0, -86.6, 0.0]
    )