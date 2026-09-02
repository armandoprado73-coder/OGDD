"""
OGDD - Nearest Neighbor Search

Provides efficient closest-point correspondence search
for large dental point sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NearestNeighborResult:
    """
    Closest target correspondences for one source set.
    """

    source_indices: np.ndarray

    target_indices: np.ndarray

    corresponding_target_points: np.ndarray

    distances: np.ndarray

    source_point_count: int

    @property
    def match_count(
        self,
    ) -> int:
        """
        Number of source points with a valid match.
        """

        return len(
            self.source_indices
        )

    @property
    def match_fraction(
        self,
    ) -> float:
        """
        Fraction of source points with a valid match.
        """

        return (
            self.match_count
            / self.source_point_count
        )


def _validated_point_set(
    points,
    name: str,
) -> np.ndarray:
    """
    Validate a non-empty Nx3 point set.
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

    if len(points) == 0:
        raise ValueError(
            f"{name} point set cannot be empty."
        )

    if not np.all(
        np.isfinite(points)
    ):
        raise ValueError(
            f"{name} points must contain finite values."
        )

    return points


def _validated_maximum_distance(
    maximum_distance,
) -> float:
    """
    Validate an optional correspondence threshold.
    """

    if isinstance(
        maximum_distance,
        (bool, np.bool_),
    ):
        raise TypeError(
            "Maximum distance must be numeric."
        )

    try:
        maximum_distance = float(
            maximum_distance
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise TypeError(
            "Maximum distance must be numeric."
        ) from error

    if (
        not np.isfinite(
            maximum_distance
        )
        or maximum_distance <= 0.0
    ):
        raise ValueError(
            "Maximum distance must be finite "
            "and greater than zero."
        )

    return maximum_distance


class NearestNeighborSearch:
    """
    Spatial index for a fixed target point set.

    The target tree is constructed once and can be
    queried repeatedly during iterative registration.
    """

    def __init__(
        self,
        target_points,
    ) -> None:
        """
        Build the spatial index.
        """

        target_points = (
            _validated_point_set(
                points=target_points,
                name="Target",
            )
        )

        self._target_points = (
            target_points.copy()
        )

        self._target_points.setflags(
            write=False
        )

        self._tree = cKDTree(
            self._target_points
        )

    @property
    def target_points(
        self,
    ) -> np.ndarray:
        """
        Return a copy of the fixed target points.
        """

        return self._target_points.copy()

    def query(
        self,
        source_points,
        maximum_distance=None,
    ) -> NearestNeighborResult:
        """
        Find the closest target point for every source.

        When ``maximum_distance`` is supplied, source
        points farther than that threshold are excluded
        from the returned correspondences.
        """

        source = _validated_point_set(
            points=source_points,
            name="Source",
        )

        distance_upper_bound = np.inf

        if maximum_distance is not None:
            distance_upper_bound = (
                _validated_maximum_distance(
                    maximum_distance
                )
            )

        distances, target_indices = (
            self._tree.query(
                source,
                k=1,
                distance_upper_bound=(
                    distance_upper_bound
                ),
            )
        )

        valid_matches = (
            np.isfinite(distances)
            & (
                target_indices
                < len(self._target_points)
            )
        )

        source_indices = np.flatnonzero(
            valid_matches
        ).astype(
            np.int64,
            copy=False,
        )

        matched_target_indices = (
            target_indices[
                valid_matches
            ].astype(
                np.int64,
                copy=False,
            )
        )

        matched_distances = (
            distances[
                valid_matches
            ].astype(
                float,
                copy=False,
            )
        )

        corresponding_target_points = (
            self._target_points[
                matched_target_indices
            ].copy()
        )

        source_indices.setflags(
            write=False
        )

        matched_target_indices.setflags(
            write=False
        )

        matched_distances.setflags(
            write=False
        )

        corresponding_target_points.setflags(
            write=False
        )

        return NearestNeighborResult(
            source_indices=source_indices,
            target_indices=(
                matched_target_indices
            ),
            corresponding_target_points=(
                corresponding_target_points
            ),
            distances=matched_distances,
            source_point_count=len(
                source
            ),
        )