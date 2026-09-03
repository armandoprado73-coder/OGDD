"""
Tests for registering a MIC record to an RC mount.
"""

import numpy as np
import pytest

from ogdd.geometry.transform import Transform
from ogdd.mesh import Mesh
from ogdd.registration.centric_relation_registration import (
    CentricRelationRegistration,
)
from ogdd.registration.occlusal_record import (
    OcclusalRecord,
)


def surface_points() -> np.ndarray:
    """
    Return an asymmetric non-degenerate surface.
    """

    return np.array(
        [
            [-4.0, -2.0, -1.0],
            [-1.0, -3.0, 2.0],
            [2.0, -2.0, -0.5],
            [4.0, 1.0, 1.0],
            [1.0, 4.0, -1.5],
            [-3.0, 3.0, 0.75],
            [0.0, 0.0, 3.0],
            [2.5, 2.5, 2.0],
            [-2.5, 0.5, -3.0],
        ]
    )


def surface_faces(
    offset: int = 0,
) -> np.ndarray:
    """
    Return connected triangular faces.
    """

    return (
        np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 5],
                [0, 5, 6],
                [0, 6, 7],
                [0, 7, 8],
            ],
            dtype=np.int32,
        )
        + offset
    )


def rigid_transform(
    axis,
    angle_degrees,
    translation,
) -> Transform:
    """
    Build one rigid rotation and translation.
    """

    rotation = Transform.rotation_about_axis(
        origin=np.zeros(3),
        axis=np.asarray(
            axis,
            dtype=float,
        ),
        angle_degrees=angle_degrees,
    )

    matrix = rotation.matrix.copy()

    matrix[:3, 3] += np.asarray(
        translation,
        dtype=float,
    )

    return Transform(
        matrix
    )


def scanner_to_mount_transform() -> Transform:
    """
    Coordinate transform from scanner to RC mount.
    """

    return rigid_transform(
        axis=[0.2, 0.5, 1.0],
        angle_degrees=1.0,
        translation=[0.06, -0.04, 0.03],
    )


def expected_rc_to_mic_transform() -> Transform:
    """
    Known clinical mandibular movement.
    """

    return rigid_transform(
        axis=[-0.4, 0.2, 1.0],
        angle_degrees=0.8,
        translation=[0.04, -0.06, 0.02],
    )


def build_registration_case(
    *,
    record_to_mount: Transform | None = None,
    rc_to_mic: Transform | None = None,
):
    """
    Build an RC mounting and a combined MIC record.

    The record may use a scanner coordinate system that
    differs from the mounting coordinate system.
    """

    if record_to_mount is None:
        record_to_mount = (
            scanner_to_mount_transform()
        )

    if rc_to_mic is None:
        rc_to_mic = (
            expected_rc_to_mic_transform()
        )

    base = surface_points()

    maxillary_mount_points = (
        base
        + np.array(
            [0.3, -0.2, 6.0]
        )
    )

    mandibular_rc_points = (
        base
        + np.array(
            [-0.4, 0.25, -6.0]
        )
    )

    mandibular_mic_mount_points = (
        rc_to_mic.apply(
            mandibular_rc_points
        )
    )

    mount_to_record = (
        record_to_mount.inverse()
    )

    maxillary_record_points = (
        mount_to_record.apply(
            maxillary_mount_points
        )
    )

    mandibular_record_points = (
        mount_to_record.apply(
            mandibular_mic_mount_points
        )
    )

    maxillary_mesh = Mesh(
        vertices=maxillary_mount_points,
        faces=surface_faces(),
    )

    mandibular_rc_mesh = Mesh(
        vertices=mandibular_rc_points,
        faces=surface_faces(),
    )

    record_vertices = np.vstack(
        [
            maxillary_record_points,
            mandibular_record_points,
        ]
    )

    record_faces = np.vstack(
        [
            surface_faces(),
            surface_faces(
                offset=len(
                    maxillary_record_points
                )
            ),
        ]
    )

    mic_record = OcclusalRecord(
        mesh=Mesh(
            vertices=record_vertices,
            faces=record_faces,
        ),
        maxillary_vertex_indices=np.arange(
            len(maxillary_record_points),
            dtype=int,
        ),
        mandibular_vertex_indices=np.arange(
            len(maxillary_record_points),
            len(record_vertices),
            dtype=int,
        ),
    )

    return (
        maxillary_mesh,
        mandibular_rc_mesh,
        mic_record,
        record_to_mount,
        rc_to_mic,
    )


def register_case(
    **kwargs,
):
    """
    Register the standard synthetic case.
    """

    (
        maxillary_mesh,
        mandibular_rc_mesh,
        mic_record,
        _,
        _,
    ) = build_registration_case()

    return CentricRelationRegistration.register(
        maxillary_mesh=maxillary_mesh,
        mandibular_rc_mesh=(
            mandibular_rc_mesh
        ),
        mic_record=mic_record,
        trim_fraction=1.0,
        sample_size=None,
        maximum_iterations=20,
        tolerance=1e-12,
        **kwargs,
    )


def test_registration_recovers_record_coordinate_transform():
    (
        maxillary_mesh,
        mandibular_rc_mesh,
        mic_record,
        expected_record_to_mount,
        _,
    ) = build_registration_case()

    result = CentricRelationRegistration.register(
        maxillary_mesh=maxillary_mesh,
        mandibular_rc_mesh=(
            mandibular_rc_mesh
        ),
        mic_record=mic_record,
        trim_fraction=1.0,
        sample_size=None,
        maximum_iterations=20,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result.record_to_mount_transform.matrix,
        expected_record_to_mount.matrix,
        atol=1e-10,
    )


def test_registration_recovers_rc_to_mic_movement():
    (
        maxillary_mesh,
        mandibular_rc_mesh,
        mic_record,
        _,
        expected_rc_to_mic,
    ) = build_registration_case()

    result = CentricRelationRegistration.register(
        maxillary_mesh=maxillary_mesh,
        mandibular_rc_mesh=(
            mandibular_rc_mesh
        ),
        mic_record=mic_record,
        trim_fraction=1.0,
        sample_size=None,
        maximum_iterations=20,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result.mandibular_rc_to_mic_transform.matrix,
        expected_rc_to_mic.matrix,
        atol=1e-10,
    )


def test_mic_to_rc_is_inverse_clinical_movement():
    result = register_case()

    identity = (
        result.mandibular_mic_to_rc_transform.matrix
        @ result.mandibular_rc_to_mic_transform.matrix
    )

    np.testing.assert_allclose(
        identity,
        np.eye(4),
        atol=1e-12,
    )


def test_scanner_coordinates_do_not_change_clinical_result():
    rc_to_mic = (
        expected_rc_to_mic_transform()
    )

    case_a = build_registration_case(
        record_to_mount=Transform.identity(),
        rc_to_mic=rc_to_mic,
    )

    case_b = build_registration_case(
        record_to_mount=(
            scanner_to_mount_transform()
        ),
        rc_to_mic=rc_to_mic,
    )

    result_a = CentricRelationRegistration.register(
        maxillary_mesh=case_a[0],
        mandibular_rc_mesh=case_a[1],
        mic_record=case_a[2],
        trim_fraction=1.0,
        sample_size=None,
        tolerance=1e-12,
    )

    result_b = CentricRelationRegistration.register(
        maxillary_mesh=case_b[0],
        mandibular_rc_mesh=case_b[1],
        mic_record=case_b[2],
        trim_fraction=1.0,
        sample_size=None,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result_a
        .mandibular_rc_to_mic_transform
        .matrix,
        result_b
        .mandibular_rc_to_mic_transform
        .matrix,
        atol=1e-10,
    )


def test_both_surface_registrations_converge():
    result = register_case()

    assert result.converged is True
    assert (
        result.maxillary_registration.converged
        is True
    )
    assert (
        result.mandibular_registration.converged
        is True
    )


def test_both_surface_registrations_have_zero_error():
    result = register_case()

    assert (
        result
        .maxillary_registration
        .root_mean_square_error
        == pytest.approx(
            0.0,
            abs=1e-10,
        )
    )

    assert (
        result
        .mandibular_registration
        .root_mean_square_error
        == pytest.approx(
            0.0,
            abs=1e-10,
        )
    )


def test_registration_does_not_change_input_geometry():
    (
        maxillary_mesh,
        mandibular_rc_mesh,
        mic_record,
        _,
        _,
    ) = build_registration_case()

    original_maxillary = (
        maxillary_mesh.vertices.copy()
    )

    original_mandibular = (
        mandibular_rc_mesh.vertices.copy()
    )

    original_record = (
        mic_record.mesh.vertices.copy()
    )

    CentricRelationRegistration.register(
        maxillary_mesh=maxillary_mesh,
        mandibular_rc_mesh=(
            mandibular_rc_mesh
        ),
        mic_record=mic_record,
        trim_fraction=1.0,
        sample_size=None,
        tolerance=1e-12,
    )

    np.testing.assert_array_equal(
        maxillary_mesh.vertices,
        original_maxillary,
    )

    np.testing.assert_array_equal(
        mandibular_rc_mesh.vertices,
        original_mandibular,
    )

    np.testing.assert_array_equal(
        mic_record.mesh.vertices,
        original_record,
    )


def test_initial_transforms_support_large_offsets():
    record_to_mount = Transform.translation(
        [20.0, -15.0, 8.0]
    )

    rc_to_mic = Transform.translation(
        [5.0, -4.0, 3.0]
    )

    (
        maxillary_mesh,
        mandibular_rc_mesh,
        mic_record,
        _,
        _,
    ) = build_registration_case(
        record_to_mount=record_to_mount,
        rc_to_mic=rc_to_mic,
    )

    result = CentricRelationRegistration.register(
        maxillary_mesh=maxillary_mesh,
        mandibular_rc_mesh=(
            mandibular_rc_mesh
        ),
        mic_record=mic_record,
        record_to_mount_initial_transform=(
            record_to_mount
        ),
        mandibular_mic_to_rc_initial_transform=(
            rc_to_mic.inverse()
        ),
        trim_fraction=1.0,
        sample_size=None,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result.record_to_mount_transform.matrix,
        record_to_mount.matrix,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.mandibular_rc_to_mic_transform.matrix,
        rc_to_mic.matrix,
        atol=1e-12,
    )


def test_non_mesh_maxillary_input_is_rejected():
    with pytest.raises(
        TypeError,
        match="Maxillary mesh",
    ):
        CentricRelationRegistration.register(
            maxillary_mesh=np.zeros((4, 3)),
            mandibular_rc_mesh=Mesh(
                vertices=surface_points(),
                faces=surface_faces(),
            ),
            mic_record=build_registration_case()[2],
        )


def test_non_mesh_mandibular_input_is_rejected():
    with pytest.raises(
        TypeError,
        match="Mandibular RC mesh",
    ):
        CentricRelationRegistration.register(
            maxillary_mesh=Mesh(
                vertices=surface_points(),
                faces=surface_faces(),
            ),
            mandibular_rc_mesh=np.zeros((4, 3)),
            mic_record=build_registration_case()[2],
        )


def test_small_maxillary_mesh_is_rejected():
    with pytest.raises(
        ValueError,
        match="three vertices",
    ):
        CentricRelationRegistration.register(
            maxillary_mesh=Mesh(
                vertices=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                )
            ),
            mandibular_rc_mesh=(
                build_registration_case()[1]
            ),
            mic_record=build_registration_case()[2],
        )


def test_invalid_mic_record_is_rejected():
    with pytest.raises(
        TypeError,
        match="OcclusalRecord",
    ):
        CentricRelationRegistration.register(
            maxillary_mesh=build_registration_case()[0],
            mandibular_rc_mesh=(
                build_registration_case()[1]
            ),
            mic_record=np.zeros((8, 3)),
        )


@pytest.mark.parametrize(
    "invalid_iterations",
    [
        0,
        -1,
        1.5,
        True,
    ],
)
def test_invalid_surface_parameters_are_forwarded(
    invalid_iterations,
):
    (
        maxillary_mesh,
        mandibular_rc_mesh,
        mic_record,
        _,
        _,
    ) = build_registration_case()

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        CentricRelationRegistration.register(
            maxillary_mesh=maxillary_mesh,
            mandibular_rc_mesh=(
                mandibular_rc_mesh
            ),
            mic_record=mic_record,
            maximum_iterations=invalid_iterations,
        )