"""
Tests for iterative closest-point surface registration.
"""

import numpy as np
import pytest

from ogdd.geometry.transform import Transform
from ogdd.registration.surface_registration import (
    SurfaceRegistration,
)


def surface_points() -> np.ndarray:
    """
    Return an asymmetric non-degenerate point surface.
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


def small_rigid_transform() -> Transform:
    """
    Build a small known rigid transformation.

    The displacement is intentionally smaller than the
    distance between neighboring source points so ICP
    begins inside the correct correspondence basin.
    """

    transform = (
        Transform.rotation_about_axis(
            origin=np.zeros(3),
            axis=np.array(
                [0.3, 0.7, 1.0]
            ),
            angle_degrees=1.5,
        )
    )

    matrix = transform.matrix.copy()

    matrix[:3, 3] += np.array(
        [0.05, -0.04, 0.03]
    )

    return Transform(
        matrix
    )


def test_identity_surface_registration():
    source = surface_points()

    result = SurfaceRegistration.align_points(
        source_points=source,
        target_points=source,
        trim_fraction=1.0,
        sample_size=None,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result.transform.matrix,
        np.eye(4),
        atol=1e-12,
    )

    assert result.converged is True
    assert result.iteration_count == 1


def test_surface_registration_recovers_known_transform():
    source = surface_points()

    expected = small_rigid_transform()

    target = expected.apply(
        source
    )

    result = SurfaceRegistration.align_points(
        source_points=source,
        target_points=target,
        trim_fraction=1.0,
        sample_size=None,
        maximum_iterations=20,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result.transform.matrix,
        expected.matrix,
        atol=1e-10,
    )


def test_exact_surface_registration_has_zero_error():
    source = surface_points()

    target = (
        small_rigid_transform()
        .apply(source)
    )

    result = SurfaceRegistration.align_points(
        source_points=source,
        target_points=target,
        trim_fraction=1.0,
        sample_size=None,
        tolerance=1e-12,
    )

    assert (
        result.root_mean_square_error
        == pytest.approx(
            0.0,
            abs=1e-10,
        )
    )

    assert (
        result.mean_error
        == pytest.approx(
            0.0,
            abs=1e-10,
        )
    )

    assert (
        result.maximum_error
        == pytest.approx(
            0.0,
            abs=1e-10,
        )
    )


def test_result_reports_registration_counts():
    source = surface_points()

    result = SurfaceRegistration.align_points(
        source_points=source,
        target_points=source,
        trim_fraction=1.0,
        sample_size=None,
    )

    assert result.point_count == len(
        source
    )

    assert (
        result.accepted_correspondence_count
        == len(source)
    )

    assert (
        len(
            result
            .iteration_root_mean_square_errors
        )
        == result.iteration_count
    )


def test_result_error_arrays_are_read_only():
    source = surface_points()

    result = SurfaceRegistration.align_points(
        source_points=source,
        target_points=source,
        trim_fraction=1.0,
        sample_size=None,
    )

    assert (
        result.nearest_neighbor_errors.flags.writeable
        is False
    )

    assert (
        result
        .iteration_root_mean_square_errors
        .flags
        .writeable
        is False
    )


def test_source_and_target_may_have_different_counts():
    source = surface_points()

    expected = small_rigid_transform()

    target = expected.apply(
        source
    )

    target = np.vstack(
        [
            target,
            [50.0, 50.0, 50.0],
            [-50.0, 50.0, -50.0],
        ]
    )

    result = SurfaceRegistration.align_points(
        source_points=source,
        target_points=target,
        trim_fraction=1.0,
        sample_size=None,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result.transform.matrix,
        expected.matrix,
        atol=1e-10,
    )


def test_trimmed_registration_ignores_far_outlier():
    base_source = surface_points()

    source = np.vstack(
        [
            base_source,
            [100.0, 100.0, 100.0],
        ]
    )

    expected = small_rigid_transform()

    target = expected.apply(
        base_source
    )

    result = SurfaceRegistration.align_points(
        source_points=source,
        target_points=target,
        trim_fraction=0.90,
        sample_size=None,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result.transform.matrix,
        expected.matrix,
        atol=1e-10,
    )

    assert (
        result.accepted_correspondence_count
        == len(base_source)
    )


def test_initial_transform_supports_large_displacement():
    source = surface_points()

    expected = Transform.translation(
        [8.0, -6.0, 4.0]
    )

    target = expected.apply(
        source
    )

    initial = Transform.translation(
        [7.95, -5.96, 3.97]
    )

    result = SurfaceRegistration.align_points(
        source_points=source,
        target_points=target,
        initial_transform=initial,
        trim_fraction=1.0,
        sample_size=None,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(
        result.transform.matrix,
        expected.matrix,
        atol=1e-10,
    )


def test_registration_does_not_change_input_points():
    source = surface_points()

    target = (
        small_rigid_transform()
        .apply(source)
    )

    original_source = source.copy()
    original_target = target.copy()

    SurfaceRegistration.align_points(
        source_points=source,
        target_points=target,
        trim_fraction=1.0,
        sample_size=None,
    )

    np.testing.assert_array_equal(
        source,
        original_source,
    )

    np.testing.assert_array_equal(
        target,
        original_target,
    )


@pytest.mark.parametrize(
    "invalid_points",
    [
        np.array(
            [0.0, 1.0, 2.0]
        ),
        np.zeros(
            (4, 2)
        ),
    ],
)
def test_invalid_source_shape_is_rejected(
    invalid_points,
):
    with pytest.raises(
        ValueError,
        match=r"shape \(N,3\)",
    ):
        SurfaceRegistration.align_points(
            source_points=invalid_points,
            target_points=surface_points(),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
    ],
)
def test_nonfinite_source_points_are_rejected(
    invalid_value,
):
    source = surface_points()
    source[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        SurfaceRegistration.align_points(
            source_points=source,
            target_points=surface_points(),
        )


def test_collinear_surface_is_rejected():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="non-collinear",
    ):
        SurfaceRegistration.align_points(
            source_points=source,
            target_points=surface_points(),
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
def test_invalid_maximum_iterations_are_rejected(
    invalid_iterations,
):
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        SurfaceRegistration.align_points(
            source_points=surface_points(),
            target_points=surface_points(),
            maximum_iterations=invalid_iterations,
        )


@pytest.mark.parametrize(
    "invalid_tolerance",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_tolerance_is_rejected(
    invalid_tolerance,
):
    with pytest.raises(
        ValueError,
        match="positive and finite",
    ):
        SurfaceRegistration.align_points(
            source_points=surface_points(),
            target_points=surface_points(),
            tolerance=invalid_tolerance,
        )


@pytest.mark.parametrize(
    "invalid_fraction",
    [
        0.0,
        -0.1,
        1.1,
        np.nan,
    ],
)
def test_invalid_trim_fraction_is_rejected(
    invalid_fraction,
):
    with pytest.raises(
        ValueError,
        match="Trim fraction",
    ):
        SurfaceRegistration.align_points(
            source_points=surface_points(),
            target_points=surface_points(),
            trim_fraction=invalid_fraction,
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_maximum_distance_is_rejected(
    invalid_distance,
):
    with pytest.raises(
        ValueError,
        match="positive and finite",
    ):
        SurfaceRegistration.align_points(
            source_points=surface_points(),
            target_points=surface_points(),
            maximum_correspondence_distance=(
                invalid_distance
            ),
        )


@pytest.mark.parametrize(
    "invalid_sample_size",
    [
        0,
        -1,
        2,
        3.5,
        True,
    ],
)
def test_invalid_sample_size_is_rejected(
    invalid_sample_size,
):
    with pytest.raises(
        ValueError,
    ):
        SurfaceRegistration.align_points(
            source_points=surface_points(),
            target_points=surface_points(),
            sample_size=invalid_sample_size,
        )


def test_invalid_initial_transform_is_rejected():
    with pytest.raises(
        TypeError,
        match="Initial transform",
    ):
        SurfaceRegistration.align_points(
            source_points=surface_points(),
            target_points=surface_points(),
            initial_transform=np.eye(4),
        )


def test_nonfinite_initial_transform_is_rejected():
    matrix = np.eye(4)
    matrix[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        SurfaceRegistration.align_points(
            source_points=surface_points(),
            target_points=surface_points(),
            initial_transform=Transform(
                matrix
            ),
        )


def test_too_few_accepted_correspondences_are_rejected():
    source = (
        surface_points()
        + np.array(
            [100.0, 100.0, 100.0]
        )
    )

    with pytest.raises(
        ValueError,
        match="three accepted correspondences",
    ):
        SurfaceRegistration.align_points(
            source_points=source,
            target_points=surface_points(),
            maximum_correspondence_distance=0.01,
            sample_size=None,
        )