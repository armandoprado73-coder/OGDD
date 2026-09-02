"""
Tests for the occlusal record builder.
"""

import numpy as np
import pytest

from ogdd.mesh import Mesh
from ogdd.registration.occlusal_record_builder import (
    OcclusalRecordBuilder,
)


def build_combined_mesh() -> Mesh:
    """
    Build two arch components and one unrelated fragment.
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
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
        ]
    )

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [8, 9, 10],
        ]
    )

    return Mesh(
        vertices=vertices,
        faces=faces,
    )


def test_builder_finds_both_seeded_arches():
    mesh = build_combined_mesh()

    record = (
        OcclusalRecordBuilder
        .from_seed_vertices(
            mesh=mesh,
            maxillary_seed_vertex=1,
            mandibular_seed_vertex=6,
        )
    )

    np.testing.assert_array_equal(
        record.maxillary_vertex_indices,
        np.array([0, 1, 2, 3]),
    )

    np.testing.assert_array_equal(
        record.mandibular_vertex_indices,
        np.array([4, 5, 6, 7]),
    )


def test_builder_ignores_unselected_fragment():
    mesh = build_combined_mesh()

    record = (
        OcclusalRecordBuilder
        .from_seed_vertices(
            mesh=mesh,
            maxillary_seed_vertex=0,
            mandibular_seed_vertex=4,
        )
    )

    selected_indices = np.concatenate(
        [
            record.maxillary_vertex_indices,
            record.mandibular_vertex_indices,
        ]
    )

    assert not np.any(
        np.isin(
            np.array([8, 9, 10]),
            selected_indices,
        )
    )


def test_seed_identity_defines_arch_identity():
    mesh = build_combined_mesh()

    record = (
        OcclusalRecordBuilder
        .from_seed_vertices(
            mesh=mesh,
            maxillary_seed_vertex=5,
            mandibular_seed_vertex=2,
        )
    )

    np.testing.assert_array_equal(
        record.maxillary_vertex_indices,
        np.array([4, 5, 6, 7]),
    )

    np.testing.assert_array_equal(
        record.mandibular_vertex_indices,
        np.array([0, 1, 2, 3]),
    )


def test_geometric_proximity_does_not_join_arches():
    mesh = build_combined_mesh()

    mesh.vertices[4] = (
        mesh.vertices[0]
    )

    record = (
        OcclusalRecordBuilder
        .from_seed_vertices(
            mesh=mesh,
            maxillary_seed_vertex=0,
            mandibular_seed_vertex=4,
        )
    )

    assert len(
        record.maxillary_vertex_indices
    ) == 4

    assert len(
        record.mandibular_vertex_indices
    ) == 4


def test_seeds_in_same_component_are_rejected():
    mesh = build_combined_mesh()

    with pytest.raises(
        ValueError,
        match="different connected components",
    ):
        (
            OcclusalRecordBuilder
            .from_seed_vertices(
                mesh=mesh,
                maxillary_seed_vertex=0,
                mandibular_seed_vertex=2,
            )
        )


def test_non_mesh_input_is_rejected():
    with pytest.raises(
        TypeError,
        match="OGDD Mesh",
    ):
        (
            OcclusalRecordBuilder
            .from_seed_vertices(
                mesh=np.zeros((4, 3)),
                maxillary_seed_vertex=0,
                mandibular_seed_vertex=1,
            )
        )


def test_empty_mesh_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        (
            OcclusalRecordBuilder
            .from_seed_vertices(
                mesh=Mesh(),
                maxillary_seed_vertex=0,
                mandibular_seed_vertex=1,
            )
        )


def test_mesh_without_faces_is_rejected():
    mesh = Mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        faces=np.empty(
            (0, 3),
            dtype=np.int32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must contain faces",
    ):
        (
            OcclusalRecordBuilder
            .from_seed_vertices(
                mesh=mesh,
                maxillary_seed_vertex=0,
                mandibular_seed_vertex=1,
            )
        )


def test_invalid_face_indices_are_rejected():
    mesh = Mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array(
            [
                [0, 1, 8],
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="outside the mesh",
    ):
        (
            OcclusalRecordBuilder
            .from_seed_vertices(
                mesh=mesh,
                maxillary_seed_vertex=0,
                mandibular_seed_vertex=1,
            )
        )


@pytest.mark.parametrize(
    "seed",
    [
        1.5,
        "1",
        True,
    ],
)
def test_noninteger_seed_is_rejected(seed):
    mesh = build_combined_mesh()

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        (
            OcclusalRecordBuilder
            .from_seed_vertices(
                mesh=mesh,
                maxillary_seed_vertex=seed,
                mandibular_seed_vertex=4,
            )
        )


@pytest.mark.parametrize(
    "seed",
    [
        -1,
        50,
    ],
)
def test_seed_outside_mesh_is_rejected(seed):
    mesh = build_combined_mesh()

    with pytest.raises(
        ValueError,
        match="outside the mesh",
    ):
        (
            OcclusalRecordBuilder
            .from_seed_vertices(
                mesh=mesh,
                maxillary_seed_vertex=seed,
                mandibular_seed_vertex=4,
            )
        )