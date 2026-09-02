"""
OGDD - Iterative Closest Point

Iteratively aligns a source point set to a fixed target
using nearest-neighbor correspondences and rigid updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ogdd.geometry.transform import Transform
from ogdd.registration.nearest_neighbor import (
    NearestNeighborResult,
    NearestNeighborSearch,
)
from ogdd.registration.rigid_registration import (
    RigidRegistration,
)


@dataclass(frozen=True)
class IterativeClosestPointResult:
    """
    Result of an iterative closest-point registration.
    """

    transform: Transform

    source_indices: np.ndarray

    target_indices: np.ndarray

    correspondence_errors: np.ndarray

    error_history: np.ndarray

    source_point_count: int

    iterations: int

    converged: bool

    initial_root_mean_square_error: float

    root_mean_square_error: float

    mean_error: float

    maximum_error: float

    @property
    def match_count(
        self,
    ) -> int:
        """
        Number of final valid correspondences.
        """

        return len(
            self.source_indices
        )

    @property
    def match_fraction(
        self,
    ) -> float:
        """
        Fraction of source points used in the final fit.
        """

        return (
            self.match_count
            / self.source_point_count
        )

    @property
    def improvement(
        self,
    ) -> float:
        """
        Reduction in root mean square error.
        """

        return (
            self.initial_root_mean_square_error
            - self.root_mean_square_error
        )


def _validated_icp_points(
    points,
    name: str,
) -> np.ndarray:
    """
    Validate one point set used by ICP.
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
            f"{name} point set must contain "
            "at least three points."
        )

    if not np.all(
        np.isfinite(points)
    ):
        raise ValueError(
            f"{name} points must contain finite values."
        )

    return points


def _validated_maximum_iterations(
    value,
) -> int:
    """
    Validate the iteration limit.
    """

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise TypeError(
            "Maximum iterations must be an integer."
        )

    value = int(
        value
    )

    if value <= 0:
        raise ValueError(
            "Maximum iterations must be greater than zero."
        )

    return value


def _validated_tolerance(
    value,
) -> float:
    """
    Validate the convergence tolerance.
    """

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        raise TypeError(
            "Convergence tolerance must be numeric."
        )

    try:
        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "Convergence tolerance must be numeric."
        ) from error

    if (
        not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(
            "Convergence tolerance must be finite "
            "and non-negative."
        )

    return value


def _validated_initial_transform(
    transform,
) -> Transform:
    """
    Validate and copy an initial rigid transform.
    """

    if transform is None:
        return Transform.identity()

    if not isinstance(
        transform,
        Transform,
    ):
        raise TypeError(
            "Initial position must be a Transform."
        )

    matrix = transform.matrix

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "Initial transformation must contain "
            "finite values."
        )

    if not np.allclose(
        matrix[3],
        np.array(
            [0.0, 0.0, 0.0, 1.0]
        ),
    ):
        raise ValueError(
            "Initial transformation must use "
            "homogeneous coordinates."
        )

    rotation = matrix[
        :3,
        :3,
    ]

    if (
        not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
        )
        or not np.isclose(
            np.linalg.det(rotation),
            1.0,
        )
    ):
        raise ValueError(
            "Initial transformation must contain "
            "only rigid rotation and translation."
        )

    return Transform(
        matrix.copy()
    )


def _require_sufficient_matches(
    matches: NearestNeighborResult,
) -> None:
    """
    Require enough matches for a rigid update.
    """

    if matches.match_count < 3:
        raise ValueError(
            "ICP requires at least three valid "
            "correspondences."
        )


def _root_mean_square(
    distances: np.ndarray,
) -> float:
    """
    Calculate root mean square correspondence error.
    """

    return float(
        np.sqrt(
            np.mean(
                distances ** 2
            )
        )
    )


def _read_only_copy(
    values,
    dtype,
) -> np.ndarray:
    """
    Copy an array and prevent accidental mutation.
    """

    copied = np.asarray(
        values,
        dtype=dtype,
    ).copy()

    copied.setflags(
        write=False
    )

    return copied


class IterativeClosestPoint:
    """
    Point-to-point rigid iterative registration.

    ICP is a local refinement method. The source must
    first be placed reasonably close to the target or an
    appropriate initial transform must be supplied.
    """

    @staticmethod
    def align(
        source_points,
        target_points,
        initial_transform: Transform | None = None,
        maximum_iterations: int = 50,
        convergence_tolerance: float = 1e-6,
        maximum_correspondence_distance=None,
    ) -> IterativeClosestPointResult:
        """
        Iteratively align source points to a fixed target.
        """

        source = _validated_icp_points(
            points=source_points,
            name="Source",
        )

        target = _validated_icp_points(
            points=target_points,
            name="Target",
        )

        maximum_iterations = (
            _validated_maximum_iterations(
                maximum_iterations
            )
        )

        convergence_tolerance = (
            _validated_tolerance(
                convergence_tolerance
            )
        )

        current_transform = (
            _validated_initial_transform(
                initial_transform
            )
        )

        search = NearestNeighborSearch(
            target
        )

        positioned_source = (
            current_transform.apply(
                source
            )
        )

        matches = search.query(
            positioned_source,
            maximum_distance=(
                maximum_correspondence_distance
            ),
        )

        _require_sufficient_matches(
            matches
        )

        initial_rmse = _root_mean_square(
            matches.distances
        )

        previous_rmse = initial_rmse

        error_history = []

        converged = False

        iterations = 0

        for iteration in range(
            1,
            maximum_iterations + 1,
        ):
            incremental = (
                RigidRegistration
                .align_corresponding_points(
                    source_points=(
                        positioned_source[
                            matches.source_indices
                        ]
                    ),
                    target_points=(
                        matches
                        .corresponding_target_points
                    ),
                )
            )

            combined_matrix = (
                incremental.transform.matrix
                @ current_transform.matrix
            )

            current_transform = Transform(
                combined_matrix
            )

            positioned_source = (
                current_transform.apply(
                    source
                )
            )

            matches = search.query(
                positioned_source,
                maximum_distance=(
                    maximum_correspondence_distance
                ),
            )

            _require_sufficient_matches(
                matches
            )

            current_rmse = (
                _root_mean_square(
                    matches.distances
                )
            )

            error_history.append(
                current_rmse
            )

            iterations = iteration

            if abs(
                previous_rmse
                - current_rmse
            ) <= convergence_tolerance:
                converged = True
                break

            previous_rmse = current_rmse

        final_errors = _read_only_copy(
            values=matches.distances,
            dtype=float,
        )

        return IterativeClosestPointResult(
            transform=Transform(
                current_transform.matrix.copy()
            ),
            source_indices=_read_only_copy(
                values=matches.source_indices,
                dtype=np.int64,
            ),
            target_indices=_read_only_copy(
                values=matches.target_indices,
                dtype=np.int64,
            ),
            correspondence_errors=(
                final_errors
            ),
            error_history=_read_only_copy(
                values=error_history,
                dtype=float,
            ),
            source_point_count=len(
                source
            ),
            iterations=iterations,
            converged=converged,
            initial_root_mean_square_error=(
                initial_rmse
            ),
            root_mean_square_error=(
                _root_mean_square(
                    final_errors
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
        )