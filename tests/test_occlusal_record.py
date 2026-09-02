"""
Tests for the combined rigid occlusal record.
"""

import numpy as np
import pytest

from ogdd.geometry.transform import Transform
from ogdd.mesh import Mesh
from ogdd.registration.occlusal_record import (
    OcclusalRecord,
)


def build_record() -> OcclusalRecord:
    """
    Build a simple combined record with separate
    maxillary and mandibular regions.
    """

    vertices = np.array(
        [
            [-2.0, -1.0, 2.0],
            [2.0, -1.0, 2.0],
            [2.0, 1.0, 2.0],
            [-2.0, 1.0, 2.0],
            [-2.0, -1.0, -2.0],
            [2.0, -1.0, -2.0],
            [2.0, 1.0, -2.0],
            [-2.0, 1.0, -2.0],
        ]
    )

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
        ]
    )

    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    mesh = Mesh(
        vertices=vertices,
        faces=faces,
        normals=normals,
        attributes={
            "quality": np.arange(
                len(vertices),
                dtype=float,
            )
        },
        metadata={
            "relation": "MIC",
        },
    )

    return OcclusalRecord(
        mesh=mesh,
        maxillary_vertex_indices=np.array(
            [0, 1, 2, 3]
        ),
        mandibular_vertex_indices=np.array(
            [4, 5, 6, 7]
        ),
    )


def test_record_keeps_one_combined_mesh():
    record = build_record()

    assert record.mesh.vertex_count == 8
    assert record.mesh.face_count == 4


def test_record_exposes_separate_registration_regions():
    record = build_record()

    assert record.maxillary_points.shape == (4, 3)
    assert record.mandibular_points.shape == (4, 3)

    assert np.all(
        record.maxillary_points[:, 2] > 0.0
    )

    assert np.all(
        record.mandibular_points[:, 2] < 0.0
    )


def test_empty_record_mesh_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        OcclusalRecord(
            mesh=Mesh(),
            maxillary_vertex_indices=np.array(
                [0]
            ),
            mandibular_vertex_indices=np.array(
                [1]
            ),
        )


def test_empty_registration_region_is_rejected():
    record = build_record()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        OcclusalRecord(
            mesh=record.mesh,
            maxillary_vertex_indices=np.array(
                [],
                dtype=int,
            ),
            mandibular_vertex_indices=np.array(
                [4, 5, 6, 7]
            ),
        )


def test_outside_vertex_indices_are_rejected():
    record = build_record()

    with pytest.raises(
        ValueError,
        match="outside the mesh",
    ):
        OcclusalRecord(
            mesh=record.mesh,
            maxillary_vertex_indices=np.array(
                [0, 1, 20]
            ),
            mandibular_vertex_indices=np.array(
                [4, 5, 6, 7]
            ),
        )


def test_overlapping_regions_are_rejected():
    record = build_record()

    with pytest.raises(
        ValueError,
        match="cannot overlap",
    ):
        OcclusalRecord(
            mesh=record.mesh,
            maxillary_vertex_indices=np.array(
                [0, 1, 2, 3]
            ),
            mandibular_vertex_indices=np.array(
                [3, 4, 5, 6]
            ),
        )


def test_duplicate_region_indices_are_rejected():
    record = build_record()

    with pytest.raises(
        ValueError,
        match="cannot contain duplicates",
    ):
        OcclusalRecord(
            mesh=record.mesh,
            maxillary_vertex_indices=np.array(
                [0, 1, 1, 2]
            ),
            mandibular_vertex_indices=np.array(
                [4, 5, 6, 7]
            ),
        )


def test_identity_position_preserves_record_geometry():
    record = build_record()

    position = record.position_at(
        Transform.identity()
    )

    np.testing.assert_allclose(
        position.mesh.vertices,
        record.mesh.vertices,
    )

    np.testing.assert_array_equal(
        position.mesh.faces,
        record.mesh.faces,
    )


def test_translation_moves_both_regions_together():
    record = build_record()

    translation = np.array(
        [3.0, -2.0, 5.0]
    )

    position = record.position_at(
        Transform.translation(
            translation
        )
    )

    np.testing.assert_allclose(
        position.maxillary_points,
        record.maxillary_points
        + translation,
    )

    np.testing.assert_allclose(
        position.mandibular_points,
        record.mandibular_points
        + translation,
    )


def test_position_does_not_change_original_mesh():
    record = build_record()

    original_vertices = (
        record.mesh.vertices.copy()
    )

    record.position_at(
        Transform.translation(
            [10.0, 20.0, 30.0]
        )
    )

    np.testing.assert_allclose(
        record.mesh.vertices,
        original_vertices,
    )


def test_rigid_position_preserves_interarch_relation():
    record = build_record()

    original_distance = np.linalg.norm(
        record.maxillary_points.mean(axis=0)
        - record.mandibular_points.mean(axis=0)
    )

    transform = Transform.rotation_about_axis(
        origin=np.array(
            [1.0, 2.0, 3.0]
        ),
        axis=np.array(
            [1.0, 1.0, 0.0]
        ),
        angle_degrees=37.0,
    )

    position = record.position_at(
        transform
    )

    positioned_distance = np.linalg.norm(
        position.maxillary_points.mean(axis=0)
        - position.mandibular_points.mean(axis=0)
    )

    assert positioned_distance == pytest.approx(
        original_distance
    )


def test_scale_is_rejected():
    record = build_record()

    with pytest.raises(
        ValueError,
        match="only rigid",
    ):
        record.position_at(
            Transform.scale(
                2.0
            )
        )


def test_reflection_is_rejected():
    record = build_record()

    matrix = np.eye(4)
    matrix[0, 0] = -1.0

    with pytest.raises(
        ValueError,
        match="only rigid",
    ):
        record.position_at(
            Transform(
                matrix
            )
        )


def test_position_preserves_mesh_information():
    record = build_record()

    position = record.position_at(
        Transform.translation(
            [1.0, 2.0, 3.0]
        )
    )

    np.testing.assert_array_equal(
        position.mesh.normals,
        record.mesh.normals,
    )

    np.testing.assert_array_equal(
        position.mesh.attributes["quality"],
        record.mesh.attributes["quality"],
    )

    assert (
        position.mesh.metadata
        == record.mesh.metadata
    )

    assert (
        position.mesh.metadata
        is not record.mesh.metadata
    )