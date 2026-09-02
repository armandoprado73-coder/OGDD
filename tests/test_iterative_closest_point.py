"""
Tests for iterative closest-point registration.
"""

import numpy as np
import pytest

from ogdd.geometry.transform import Transform
from ogdd.registration.iterative_closest_point import (
    IterativeClosestPoint,
)


def asymmetric_point_cloud() -> np.ndarray:
    """
    Build a spaced asymmetric 3D point cloud.
    """

    points = []

    for x in np.linspace(
        -8.0,
        8.0,
        5,
    ):
        for y in np.linspace(
            -6.0,
            6.0,
            5,
        ):
            for z in np.linspace(
                -3.0,
                3.0,
                4,
            ):
                points.append(
                    [
                        x,
                        y + 0.03 * x ** 2,
                        z + 0.02 * x * y,
                    ]
                )

    return np.asarray(
        points,
        dtype=float,
    )


def rigid_transform(
    angle_degrees: float,
    translation,
) -> Transform:
    """
    Build a known rotation followed by translation.
    """

    rotation = Transform.rotation_about_axis(
        origin=np.zeros(3),
        axis=np.array(
            [0.4, 0.8, 0.2]
        ),
        angle_degrees=angle_degrees,
    )

    matrix = rotation.matrix.copy()

    matrix[:3, 3] = np.asarray(
        translation,
        dtype=float,
    )

    return Transform(
        matrix
    )


def small_transform() -> Transform:
    """
    Build a transform suitable for unassisted ICP.
    """

    return rigid_transform(
        angle_degrees=1.0,
        translation=[
            0.10,
            -0.08,
            0.06,
        ],
    )


def test_identity_registration_converges():
    source = asymmetric_point_cloud()

    result = IterativeClosestPoint.align(
        source_points=source,
        target_points=source,
    )

    assert result.converged is True
    assert result.iterations == 1

    np.testing.assert_allclose(
        result.transform.matrix,
        np.eye(4),
        atol=1e-12,
    )


def test_icp_recovers_small_rigid_transform():
    source = asymmetric_point_cloud()

    expected = small_transform()

    target = expected.apply(
        source
    )

    result = IterativeClosestPoint.align(
        source_points=source,
        target_points=target,
        convergence_tolerance=1e-10,
    )

    assert result.converged is True

    np.testing.assert_allclose(
        result.transform.matrix,
        expected.matrix,
        atol=1e-10,
    )


def test_initial_position_enables_large_registration():
    source = asymmetric_point_cloud()

    expected = rigid_transform(
        angle_degrees=28.0,
        translation=[
            20.0,
            -15.0,
            8.0,
        ],
    )

    approximate = rigid_transform(
        angle_degrees=27.5,
        translation=[
            20.1,
            -15.1,
            8.05,
        ],
    )

    target = expected.apply(
        source
    )

    result = IterativeClosestPoint.align(
        source_points=source,
        target_points=target,
        initial_transform=approximate,
        maximum_correspondence_distance=1.0,
        convergence_tolerance=1e-10,
    )

    assert result.converged is True

    np.testing.assert_allclose(
        result.transform.matrix,
        expected.matrix,
        atol=1e-10,
    )


def test_exact_fit_reports_near_zero_error():
    source = asymmetric_point_cloud()

    target = small_transform().apply(
        source
    )

    result = IterativeClosestPoint.align(
        source_points=source,
        target_points=target,
        convergence_tolerance=1e-10,
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

    assert result.improvement > 0.0


def test_result_reports_complete_matches():
    source = asymmetric_point_cloud()

    target = small_transform().apply(
        source
    )

    result = IterativeClosestPoint.align(
        source_points=source,
        target_points=target,
    )

    assert result.source_point_count == len(
        source
    )

    assert result.match_count == len(
        source
    )

    assert result.match_fraction == pytest.approx(
        1.0
    )


def test_correspondence_limit_excludes_outlier():
    base_source = asymmetric_point_cloud()

    source = np.vstack(
        [
            base_source,
            np.array(
                [
                    [100.0, 100.0, 100.0]
                ]
            ),
        ]
    )

    target = small_transform().apply(
        base_source
    )

    result = IterativeClosestPoint.align(
        source_points=source,
        target_points=target,
        maximum_correspondence_distance=1.0,
        convergence_tolerance=1e-10,
    )

    assert result.match_count == len(
        base_source
    )

    assert result.match_fraction == pytest.approx(
        len(base_source)
        / len(source)
    )

    assert len(source) - 1 not in (
        result.source_indices
    )


def test_insufficient_correspondences_are_rejected():
    source = (
        asymmetric_point_cloud()
        + 1000.0
    )

    target = asymmetric_point_cloud()

    with pytest.raises(
        ValueError,
        match="at least three",
    ):
        IterativeClosestPoint.align(
            source_points=source,
            target_points=target,
            maximum_correspondence_distance=0.1,
        )


def test_iteration_limit_is_reported():
    source = asymmetric_point_cloud()

    target = small_transform().apply(
        source
    )

    result = IterativeClosestPoint.align(
        source_points=source,
        target_points=target,
        maximum_iterations=1,
        convergence_tolerance=1e-12,
    )

    assert result.iterations == 1
    assert result.converged is False

    assert (
        result.root_mean_square_error
        == pytest.approx(
            0.0,
            abs=1e-10,
        )
    )


def test_icp_does_not_change_input_points():
    source = asymmetric_point_cloud()

    target = small_transform().apply(
        source
    )

    original_source = source.copy()
    original_target = target.copy()

    IterativeClosestPoint.align(
        source_points=source,
        target_points=target,
    )

    np.testing.assert_array_equal(
        source,
        original_source,
    )

    np.testing.assert_array_equal(
        target,
        original_target,
    )


def test_result_arrays_are_read_only():
    source = asymmetric_point_cloud()

    target = small_transform().apply(
        source
    )

    result = IterativeClosestPoint.align(
        source_points=source,
        target_points=target,
    )

    with pytest.raises(
        ValueError,
    ):
        result.error_history[0] = 100.0

    with pytest.raises(
        ValueError,
    ):
        result.correspondence_errors[0] = 100.0


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
        IterativeClosestPoint.align(
            source_points=invalid_points,
            target_points=(
                asymmetric_point_cloud()
            ),
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
        IterativeClosestPoint.align(
            source_points=points,
            target_points=points,
        )


def test_nonfinite_points_are_rejected():
    source = asymmetric_point_cloud()

    source[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        IterativeClosestPoint.align(
            source_points=source,
            target_points=(
                asymmetric_point_cloud()
            ),
        )


@pytest.mark.parametrize(
    "maximum_iterations",
    [
        0,
        -1,
    ],
)
def test_nonpositive_iteration_limit_is_rejected(
    maximum_iterations,
):
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        IterativeClosestPoint.align(
            source_points=(
                asymmetric_point_cloud()
            ),
            target_points=(
                asymmetric_point_cloud()
            ),
            maximum_iterations=(
                maximum_iterations
            ),
        )


@pytest.mark.parametrize(
    "maximum_iterations",
    [
        1.5,
        True,
    ],
)
def test_noninteger_iteration_limit_is_rejected(
    maximum_iterations,
):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        IterativeClosestPoint.align(
            source_points=(
                asymmetric_point_cloud()
            ),
            target_points=(
                asymmetric_point_cloud()
            ),
            maximum_iterations=(
                maximum_iterations
            ),
        )


@pytest.mark.parametrize(
    "tolerance",
    [
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_numeric_tolerance_is_rejected(
    tolerance,
):
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        IterativeClosestPoint.align(
            source_points=(
                asymmetric_point_cloud()
            ),
            target_points=(
                asymmetric_point_cloud()
            ),
            convergence_tolerance=tolerance,
        )


@pytest.mark.parametrize(
    "tolerance",
    [
        "tight",
        True,
    ],
)
def test_nonnumeric_tolerance_is_rejected(
    tolerance,
):
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        IterativeClosestPoint.align(
            source_points=(
                asymmetric_point_cloud()
            ),
            target_points=(
                asymmetric_point_cloud()
            ),
            convergence_tolerance=tolerance,
        )


def test_invalid_initial_transform_type_is_rejected():
    with pytest.raises(
        TypeError,
        match="must be a Transform",
    ):
        IterativeClosestPoint.align(
            source_points=(
                asymmetric_point_cloud()
            ),
            target_points=(
                asymmetric_point_cloud()
            ),
            initial_transform=np.eye(4),
        )


def test_nonrigid_initial_transform_is_rejected():
    with pytest.raises(
        ValueError,
        match="only rigid",
    ):
        IterativeClosestPoint.align(
            source_points=(
                asymmetric_point_cloud()
            ),
            target_points=(
                asymmetric_point_cloud()
            ),
            initial_transform=Transform.scale(
                2.0
            ),
        )


def test_collinear_correspondences_are_rejected():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="non-collinear",
    ):
        IterativeClosestPoint.align(
            source_points=points,
            target_points=points,
        )