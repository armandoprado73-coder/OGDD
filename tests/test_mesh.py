"""
OGDD Mesh Tests

Initial unit tests for the Mesh core structure.
"""

import numpy as np
import pytest

from ogdd.mesh import Mesh



def test_empty_mesh_creation():
    """
    Test that an empty mesh can be created.
    """

    mesh = Mesh()

    assert mesh.vertex_count == 0
    assert mesh.face_count == 0



def test_triangle_mesh_creation():
    """
    Test creation of a simple triangular mesh.
    """

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )


    faces = np.array(
        [
            [0, 1, 2]
        ]
    )


    mesh = Mesh(
        vertices=vertices,
        faces=faces
    )


    assert mesh.vertex_count == 3
    assert mesh.face_count == 1



def test_invalid_vertex_shape():
    """
    Test invalid vertex dimensions.
    """

    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0]
        ]
    )


    with pytest.raises(ValueError):

        Mesh(vertices=vertices)



def test_attribute_storage():
    """
    Test mesh attribute system.
    """

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )


    mesh = Mesh(vertices=vertices)


    curvature = np.array(
        [
            0.1,
            0.2,
            0.3
        ]
    )


    mesh.add_attribute(
        "curvature",
        curvature
    )


    assert "curvature" in mesh.attributes

    assert np.array_equal(
        mesh.attributes["curvature"],
        curvature
    )
