"""
OGDD - Rigid Registration

Calculates the best rigid transformation between
corresponding 3D point sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ogdd.geometry.transform import Transform


@dataclass(frozen=True)
class RigidRegistrationResult:
    """
    Result of a rigid corresponding-point registration.

    Errors are expressed in the same units as the input
    points, normally millimeters in OGDD.
    """

    transform: Transform

    correspondence_errors: np.ndarray

    root_mean_square_error: float

    mean_error: float

    maximum_error: float

    @property
    def point_count(
        self,
    ) -> int:
        """
        Number of registered point correspondences.
        """

        return len(
            self.correspondence_errors
        )


def _validated_points(
    points,
    name: str,
) -> np.ndarray:
    """
    Validate one Nx3 point set.
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

    if not np.all(
        np.isfinite(points)
    ):
        raise ValueError(
            f"{name} points must contain finite values."
        )

    return points


def _validate_spatial_information(
    points: np.ndarray,
    name: str,
) -> None:
    """
    Require enough geometry to determine a unique
    rigid transformation.
    """

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


class RigidRegistration:
    """
    Best-fit rigid registration for corresponding points.

    The implementation uses the Kabsch algorithm and
    always produces a proper rotation with determinant
    positive one.
    """

    @staticmethod
    def align_corresponding_points(
        source_points,
        target_points,
    ) -> RigidRegistrationResult:
        """
        Align source points to their corresponding targets.

        Parameters
        ----------
        source_points:
            Nx3 points that will be transformed.

        target_points:
            Nx3 fixed reference points. Row ``i`` must
            correspond to row ``i`` in ``source_points``.
        """

        source = _validated_points(
            points=source_points,
            name="Source",
        )

        target = _validated_points(
            points=target_points,
            name="Target",
        )

        if len(source) != len(target):
            raise ValueError(
                "Source and target must contain "
                "the same number of points."
            )

        if len(source) < 3:
            raise ValueError(
                "Rigid registration requires at least "
                "three point correspondences."
            )

        _validate_spatial_information(
            points=source,
            name="Source",
        )

        _validate_spatial_information(
            points=target,
            name="Target",
        )

        source_centroid = source.mean(
            axis=0
        )

        target_centroid = target.mean(
            axis=0
        )

        centered_source = (
            source
            - source_centroid
        )

        centered_target = (
            target
            - target_centroid
        )

        covariance = (
            centered_source.T
            @ centered_target
        )

        left_vectors, _, right_vectors_transposed = (
            np.linalg.svd(
                covariance
            )
        )

        rotation = (
            right_vectors_transposed.T
            @ left_vectors.T
        )

        if np.linalg.det(
            rotation
        ) < 0.0:
            right_vectors_transposed[
                -1,
                :,
            ] *= -1.0

            rotation = (
                right_vectors_transposed.T
                @ left_vectors.T
            )

        translation = (
            target_centroid
            - rotation @ source_centroid
        )

        matrix = np.eye(4)

        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation

        transform = Transform(
            matrix
        )

        aligned_source = transform.apply(
            source
        )

        errors = np.linalg.norm(
            aligned_source - target,
            axis=1,
        )

        errors.setflags(
            write=False
        )

        return RigidRegistrationResult(
            transform=transform,
            correspondence_errors=errors,
            root_mean_square_error=float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            ),
            mean_error=float(
                np.mean(
                    errors
                )
            ),
            maximum_error=float(
                np.max(
                    errors
                )
            ),
        )