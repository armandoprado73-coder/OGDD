"""
OGDD STL Reader Tests

Tests for importing STL geometry
into the OGDD Mesh structure.
"""

from pathlib import Path

import numpy as np

from ogdd.io.stl import STLReader



def create_test_stl(
    filename: Path
) -> None:
    """
    Create a minimal ASCII STL file.

    Geometry:

        triangle

        (0,0,0)
        (1,0,0)
        (0,1,0)
    """

    content = """
solid test

facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet

endsolid test
"""


    filename.write_text(
        content.strip()
    )



def test_ascii_stl_loading(
    tmp_path
):
    """
    Test ASCII STL import.
    """

    stl_file = tmp_path / "test.stl"


    create_test_stl(
        stl_file
    )


    mesh = STLReader.read(
        stl_file
    )


    assert mesh.vertex_count == 3

    assert mesh.face_count == 1



def test_stl_generates_valid_mesh(
    tmp_path
):
    """
    Test resulting mesh data.
    """

    stl_file = tmp_path / "triangle.stl"


    create_test_stl(
        stl_file
    )


    mesh = STLReader.read(
        stl_file
    )


    assert mesh.vertices.shape == (
        3,
        3
    )


    assert mesh.faces.shape == (
        1,
        3
    )


    assert np.array_equal(
        mesh.faces[0],
        [0,1,2]
    )
