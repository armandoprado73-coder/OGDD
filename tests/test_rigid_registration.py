"""
Tests for corresponding-point rigid registration.
"""

import numpy as np
import pytest

from ogdd.geometry.transform import Transform
from ogdd.registration.rigid_registration import (
    RigidRegistration,
)


def source_points() -> np.ndarray:
    """
    Return an asymmetric non-degenerate 3D point set.
    """

    return np.array(
        [
            [-3.0, -1.0, 0.0],
            [2.0, -2.0, 0.5],
            [3.0, 1.0, 1.0],
            [0.0, 4.0, 0.25],
            [-2.0, 2.0, -1.0],
            [0.5, 0.5, 3.0],
        ]
    )


def arbitrary_rigid_transform() -> Transform:
    """
    Build a known rotation and translation.
    """

    rotation = Transform.rotation_about_axis(
        origin=np.array(
            [1.0, -2.0, 0.5]
        ),
        axis=np.array(
            [0.5, 1.0, 0.25]
        ),
        angle_degrees=32.0,
    )

    matrix = rotation.matrix.copy()

    matrix[:3, 3] += np.array(
        [4.0, -3.0, 2.0]
    )

    return Transform(
        matrix
    )


def test_identity_registration():
    source = source_points()

    result = (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=source,
        )
    )

    np.testing.assert_allclose(
        result.transform.matrix,
        np.eye(4),
        atol=1e-12,
    )


def test_registration_recovers_translation():
    source = source_points()

    expected = Transform.translation(
        [5.0, -4.0, 3.0]
    )

    target = expected.apply(
        source
    )

    result = (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=target,
        )
    )

    np.testing.assert_allclose(
        result.transform.matrix,
        expected.matrix,
        atol=1e-12,
    )


def test_registration_recovers_arbitrary_rigid_transform():
    source = source_points()

    expected = (
        arbitrary_rigid_transform()
    )

    target = expected.apply(
        source
    )

    result = (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=target,
        )
    )

    np.testing.assert_allclose(
        result.transform.matrix,
        expected.matrix,
        atol=1e-12,
    )


def test_exact_registration_aligns_every_point():
    source = source_points()

    expected = (
        arbitrary_rigid_transform()
    )

    target = expected.apply(
        source
    )

    result = (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=target,
        )
    )

    aligned = result.transform.apply(
        source
    )

    np.testing.assert_allclose(
        aligned,
        target,
        atol=1e-12,
    )


def test_exact_registration_has_zero_error():
    source = source_points()

    target = (
        arbitrary_rigid_transform()
        .apply(source)
    )

    result = (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=target,
        )
    )

    assert (
        result.root_mean_square_error
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.mean_error
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.maximum_error
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


def test_result_reports_correspondence_count():
    source = source_points()

    result = (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=source,
        )
    )

    assert result.point_count == len(
        source
    )


def test_noisy_registration_metrics_are_consistent():
    source = source_points()

    target = (
        arbitrary_rigid_transform()
        .apply(source)
    )

    target = target + np.array(
        [
            [0.01, 0.00, 0.00],
            [-0.01, 0.01, 0.00],
            [0.00, -0.01, 0.01],
            [0.01, 0.00, -0.01],
            [0.00, 0.01, 0.00],
            [-0.01, 0.00, 0.01],
        ]
    )

    result = (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=target,
        )
    )

    expected_rmse = np.sqrt(
        np.mean(
            result.correspondence_errors ** 2
        )
    )

    assert (
        result.root_mean_square_error
        == pytest.approx(
            expected_rmse
        )
    )

    assert result.mean_error == pytest.approx(
        np.mean(
            result.correspondence_errors
        )
    )

    assert (
        result.maximum_error
        == pytest.approx(
            np.max(
                result.correspondence_errors
            )
        )
    )


def test_reflection_is_not_returned_as_rotation():
    source = source_points()

    target = source.copy()
    target[:, 0] *= -1.0

    result = (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=target,
        )
    )

    determinant = np.linalg.det(
        result.transform.matrix[
            :3,
            :3,
        ]
    )

    assert determinant == pytest.approx(
        1.0
    )

    assert result.maximum_error > 0.0


def test_registration_does_not_change_input_points():
    source = source_points()

    target = (
        arbitrary_rigid_transform()
        .apply(source)
    )

    original_source = source.copy()
    original_target = target.copy()

    (
        RigidRegistration
        .align_corresponding_points(
            source_points=source,
            target_points=target,
        )
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
def test_invalid_point_shape_is_rejected(
    invalid_points,
):
    with pytest.raises(
        ValueError,
        match=r"shape \(N,3\)",
    ):
        (
            RigidRegistration
            .align_corresponding_points(
                source_points=invalid_points,
                target_points=source_points(),
            )
        )


def test_different_point_counts_are_rejected():
    source = source_points()

    with pytest.raises(
        ValueError,
        match="same number",
    ):
        (
            RigidRegistration
            .align_corresponding_points(
                source_points=source,
                target_points=source[:-1],
            )
        )


def test_fewer_than_three_points_are_rejected():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="at least three",
    ):
        (
            RigidRegistration
            .align_corresponding_points(
                source_points=points,
                target_points=points,
            )
        )


def test_collinear_source_points_are_rejected():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Source points",
    ):
        (
            RigidRegistration
            .align_corresponding_points(
                source_points=source,
                target_points=target,
            )
        )


def test_collinear_target_points_are_rejected():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Target points",
    ):
        (
            RigidRegistration
            .align_corresponding_points(
                source_points=source,
                target_points=target,
            )
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
    ],
)
def test_nonfinite_points_are_rejected(
    invalid_value,
):
    source = source_points()
    source[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        (
            RigidRegistration
            .align_corresponding_points(
                source_points=source,
                target_points=source_points(),
            )
        )