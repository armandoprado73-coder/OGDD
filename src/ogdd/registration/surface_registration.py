"""
OGDD - Surface Registration

Aligns partially overlapping 3D point surfaces using
iterative closest-point registration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ogdd.geometry.transform import Transform
from ogdd.registration.rigid_registration import (
    RigidRegistration,
)


@dataclass(frozen=True)
class SurfaceRegistrationResult:
    """
    Result of an iterative surface registration.

    Errors are nearest-neighbor distances expressed in
    the same units as the input points, normally
    millimeters in OGDD.
    """

    transform: Transform

    nearest_neighbor_errors: np.ndarray

    iteration_root_mean_square_errors: np.ndarray

    root_mean_square_error: float

    mean_error: float

    maximum_error: float

    iteration_count: int

    accepted_correspondence_count: int

    converged: bool

    @property
    def point_count(
        self,
    ) -> int:
        """
        Number of evaluated source points.
        """

        return len(
            self.nearest_neighbor_errors
        )


def _validated_surface_points(
    points,
    name: str,
) -> np.ndarray:
    """
    Validate one surface point cloud.
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            f"{name} points must have shape (N,3)."
        )

    if len(points) < 3:
        raise ValueError(
            f"{name} surface must contain at least "
            "three points."
        )

    if not np.all(
        np.isfinite(points)
    ):
        raise ValueError(
            f"{name} points must contain finite values."
        )

    centered = (
        points
        - points.mean(axis=0)
    )

    if np.linalg.matrix_rank(
        centered
    ) < 2:
        raise ValueError(
            f"{name} points must contain at least "
            "three non-collinear positions."
        )

    return points


def _validated_positive_integer(
    value,
    name: str,
) -> int:
    """
    Validate a positive integer parameter.
    """

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return int(value)


def _validated_positive_float(
    value,
    name: str,
) -> float:
    """
    Validate a positive finite floating-point value.
    """

    value = float(value)

    if (
        not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    return value


def _sample_source_points(
    points: np.ndarray,
    sample_size: int | None,
) -> np.ndarray:
    """
    Select a deterministic source-point sample.
    """

    if sample_size is None:
        return points

    sample_size = _validated_positive_integer(
        value=sample_size,
        name="Sample size",
    )

    if sample_size < 3:
        raise ValueError(
            "Sample size must be at least three."
        )

    if sample_size >= len(points):
        return points

    indices = np.linspace(
        start=0,
        stop=len(points) - 1,
        num=sample_size,
        dtype=int,
    )

    return points[
        indices
    ]


def _accepted_correspondences(
    distances: np.ndarray,
    maximum_correspondence_distance: float | None,
    trim_fraction: float,
) -> np.ndarray:
    """
    Select usable nearest-neighbor correspondences.
    """

    accepted = np.flatnonzero(
        np.isfinite(
            distances
        )
    )

    if maximum_correspondence_distance is not None:
        accepted = accepted[
            distances[accepted]
            <= maximum_correspondence_distance
        ]

    if len(accepted) < 3:
        raise ValueError(
            "Surface registration requires at least "
            "three accepted correspondences."
        )

    keep_count = max(
        3,
        int(
            np.floor(
                len(accepted)
                * trim_fraction
            )
        ),
    )

    if keep_count < len(accepted):
        local_order = np.argpartition(
            distances[accepted],
            keep_count - 1,
        )[
            :keep_count
        ]

        accepted = accepted[
            local_order
        ]

    return accepted


class SurfaceRegistration:
    """
    Point-to-point iterative closest-point registration.

    The source surface is moved onto the fixed target
    surface. Each iteration finds nearest neighbors,
    rejects distant correspondences, and delegates the
    rigid best-fit step to ``RigidRegistration``.
    """

    @staticmethod
    def align_points(
        source_points,
        target_points,
        *,
        initial_transform: Transform | None = None,
        maximum_iterations: int = 50,
        tolerance: float = 1e-6,
        maximum_correspondence_distance: float | None = None,
        trim_fraction: float = 0.90,
        sample_size: int | None = 10000,
    ) -> SurfaceRegistrationResult:
        """
        Align a moving source surface to a fixed target.

        Parameters
        ----------
        source_points:
            Moving Nx3 surface points.

        target_points:
            Fixed Mx3 surface points.

        initial_transform:
            Optional initial source-to-target estimate.

        maximum_iterations:
            Maximum number of closest-point iterations.

        tolerance:
            Minimum change in sampled RMSE required to
            continue iterating.

        maximum_correspondence_distance:
            Optional maximum accepted nearest-neighbor
            distance.

        trim_fraction:
            Fraction of the closest correspondences kept
            during each iteration.

        sample_size:
            Maximum number of source points used during
            iteration. All source points are evaluated
            in the final metrics.
        """

        source = _validated_surface_points(
            points=source_points,
            name="Source",
        )

        target = _validated_surface_points(
            points=target_points,
            name="Target",
        )

        maximum_iterations = (
            _validated_positive_integer(
                value=maximum_iterations,
                name="Maximum iterations",
            )
        )

        tolerance = _validated_positive_float(
            value=tolerance,
            name="Tolerance",
        )

        trim_fraction = float(
            trim_fraction
        )

        if (
            not np.isfinite(trim_fraction)
            or trim_fraction <= 0.0
            or trim_fraction > 1.0
        ):
            raise ValueError(
                "Trim fraction must be greater than "
                "zero and at most one."
            )

        if maximum_correspondence_distance is not None:
            maximum_correspondence_distance = (
                _validated_positive_float(
                    value=maximum_correspondence_distance,
                    name=(
                        "Maximum correspondence "
                        "distance"
                    ),
                )
            )

        if initial_transform is None:
            current_transform = (
                Transform.identity()
            )
        elif isinstance(
            initial_transform,
            Transform,
        ):
            current_transform = Transform(
                initial_transform.matrix.copy()
            )
        else:
            raise TypeError(
                "Initial transform must be a Transform."
            )

        if not np.all(
            np.isfinite(
                current_transform.matrix
            )
        ):
            raise ValueError(
                "Initial transform must contain "
                "finite values."
            )

        sampled_source = _sample_source_points(
            points=source,
            sample_size=sample_size,
        )

        target_tree = cKDTree(
            target
        )

        history = []

        previous_error = None

        accepted_count = 0

        converged = False

        for _ in range(
            maximum_iterations
        ):
            moved_source = (
                current_transform.apply(
                    sampled_source
                )
            )

            distances, target_indices = (
                target_tree.query(
                    moved_source,
                    k=1,
                    workers=1,
                )
            )

            accepted = (
                _accepted_correspondences(
                    distances=distances,
                    maximum_correspondence_distance=(
                        maximum_correspondence_distance
                    ),
                    trim_fraction=trim_fraction,
                )
            )

            accepted_count = len(
                accepted
            )

            incremental_result = (
                RigidRegistration
                .align_corresponding_points(
                    source_points=moved_source[
                        accepted
                    ],
                    target_points=target[
                        target_indices[
                            accepted
                        ]
                    ],
                )
            )

            combined_matrix = (
                incremental_result
                .transform
                .matrix
                @ current_transform.matrix
            )

            current_transform = Transform(
                combined_matrix
            )

            evaluated_source = (
                current_transform.apply(
                    sampled_source
                )
            )

            evaluated_distances, _ = (
                target_tree.query(
                    evaluated_source,
                    k=1,
                    workers=1,
                )
            )

            evaluated_accepted = (
                _accepted_correspondences(
                    distances=evaluated_distances,
                    maximum_correspondence_distance=(
                        maximum_correspondence_distance
                    ),
                    trim_fraction=trim_fraction,
                )
            )

            accepted_count = len(
                evaluated_accepted
            )

            current_error = float(
                np.sqrt(
                    np.mean(
                        evaluated_distances[
                            evaluated_accepted
                        ]
                        ** 2
                    )
                )
            )

            history.append(
                current_error
            )

            if current_error <= tolerance:
                converged = True
                break

            if (
                previous_error is not None
                and abs(
                    previous_error
                    - current_error
                )
                <= tolerance
            ):
                converged = True
                break

            previous_error = current_error

        aligned_source = (
            current_transform.apply(
                source
            )
        )

        final_errors, _ = (
            target_tree.query(
                aligned_source,
                k=1,
                workers=1,
            )
        )

        final_errors = np.asarray(
            final_errors,
            dtype=float,
        )

        history_array = np.asarray(
            history,
            dtype=float,
        )

        final_errors.setflags(
            write=False
        )

        history_array.setflags(
            write=False
        )

        return SurfaceRegistrationResult(
            transform=current_transform,
            nearest_neighbor_errors=final_errors,
            iteration_root_mean_square_errors=(
                history_array
            ),
            root_mean_square_error=float(
                np.sqrt(
                    np.mean(
                        final_errors ** 2
                    )
                )
            ),
            mean_error=float(
                np.mean(
                    final_errors
                )
            ),
            maximum_error=float(
                np.max(
                    final_errors
                )
            ),
            iteration_count=len(
                history_array
            ),
            accepted_correspondence_count=(
                accepted_count
            ),
            converged=converged,
        )