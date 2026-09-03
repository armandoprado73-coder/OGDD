"""
OGDD - Centric Relation Registration

Registers a combined maximum-intercuspation record
against a fixed maxillary model and a mandibular model
mounted in centric relation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ogdd.geometry.transform import Transform
from ogdd.mesh import Mesh
from ogdd.registration.occlusal_record import (
    OcclusalRecord,
)
from ogdd.registration.surface_registration import (
    SurfaceRegistration,
    SurfaceRegistrationResult,
)


@dataclass(frozen=True)
class CentricRelationRegistrationResult:
    """
    Result of registering a MIC record to an RC mount.

    ``record_to_mount_transform`` removes the coordinate
    difference between the combined record and the fixed
    maxillary model.

    ``mandibular_mic_to_rc_transform`` describes how the
    anchored MIC mandibular surface returns to RC.

    ``mandibular_rc_to_mic_transform`` is the clinically
    useful inverse movement from RC to MIC.
    """

    record_to_mount_transform: Transform

    mandibular_mic_to_rc_transform: Transform

    mandibular_rc_to_mic_transform: Transform

    maxillary_registration: SurfaceRegistrationResult

    mandibular_registration: SurfaceRegistrationResult

    @property
    def converged(
        self,
    ) -> bool:
        """
        Whether both surface registrations converged.
        """

        return bool(
            self.maxillary_registration.converged
            and self.mandibular_registration.converged
        )


def _validated_mesh(
    mesh,
    name: str,
) -> Mesh:
    """
    Validate a registration mesh.
    """

    if not isinstance(
        mesh,
        Mesh,
    ):
        raise TypeError(
            f"{name} must be an OGDD Mesh."
        )

    if mesh.vertex_count < 3:
        raise ValueError(
            f"{name} must contain at least "
            "three vertices."
        )

    return mesh


class CentricRelationRegistration:
    """
    Register a MIC occlusal record to an RC mounting.

    The maxillary region is aligned first because the
    maxilla is the fixed anatomical reference. The same
    rigid transform is applied to the complete combined
    record, preserving its interarch relationship.

    The anchored mandibular MIC region is then aligned
    back to the mandibular RC model. Its inverse is the
    mandibular movement from RC to MIC.
    """

    @staticmethod
    def register(
        maxillary_mesh,
        mandibular_rc_mesh,
        mic_record,
        *,
        record_to_mount_initial_transform: (
            Transform | None
        ) = None,
        mandibular_mic_to_rc_initial_transform: (
            Transform | None
        ) = None,
        maximum_iterations: int = 50,
        tolerance: float = 1e-6,
        maximum_correspondence_distance: (
            float | None
        ) = None,
        trim_fraction: float = 0.90,
        sample_size: int | None = 10000,
    ) -> CentricRelationRegistrationResult:
        """
        Register one combined MIC record to an RC mount.

        Parameters
        ----------
        maxillary_mesh:
            Fixed maxillary reference mesh.

        mandibular_rc_mesh:
            Mandibular mesh mounted in centric relation.

        mic_record:
            Combined maxillary-mandibular record in
            maximum intercuspation.

        record_to_mount_initial_transform:
            Optional starting estimate for aligning the
            complete record to the maxillary reference.

        mandibular_mic_to_rc_initial_transform:
            Optional starting estimate for aligning the
            anchored MIC mandibular surface back to RC.

        maximum_iterations:
            Maximum iterations for each surface
            registration.

        tolerance:
            RMSE convergence tolerance.

        maximum_correspondence_distance:
            Optional maximum nearest-neighbor distance.

        trim_fraction:
            Fraction of closest correspondences retained
            during each iteration.

        sample_size:
            Maximum source points used in each iterative
            calculation.
        """

        maxillary_mesh = _validated_mesh(
            mesh=maxillary_mesh,
            name="Maxillary mesh",
        )

        mandibular_rc_mesh = _validated_mesh(
            mesh=mandibular_rc_mesh,
            name="Mandibular RC mesh",
        )

        if not isinstance(
            mic_record,
            OcclusalRecord,
        ):
            raise TypeError(
                "MIC record must be an "
                "OcclusalRecord."
            )

        maxillary_registration = (
            SurfaceRegistration.align_points(
                source_points=(
                    mic_record.maxillary_points
                ),
                target_points=(
                    maxillary_mesh.vertices
                ),
                initial_transform=(
                    record_to_mount_initial_transform
                ),
                maximum_iterations=(
                    maximum_iterations
                ),
                tolerance=tolerance,
                maximum_correspondence_distance=(
                    maximum_correspondence_distance
                ),
                trim_fraction=trim_fraction,
                sample_size=sample_size,
            )
        )

        positioned_record = (
            mic_record.position_at(
                maxillary_registration.transform
            )
        )

        mandibular_registration = (
            SurfaceRegistration.align_points(
                source_points=(
                    positioned_record
                    .mandibular_points
                ),
                target_points=(
                    mandibular_rc_mesh.vertices
                ),
                initial_transform=(
                    mandibular_mic_to_rc_initial_transform
                ),
                maximum_iterations=(
                    maximum_iterations
                ),
                tolerance=tolerance,
                maximum_correspondence_distance=(
                    maximum_correspondence_distance
                ),
                trim_fraction=trim_fraction,
                sample_size=sample_size,
            )
        )

        mandibular_mic_to_rc_transform = (
            mandibular_registration.transform
        )

        mandibular_rc_to_mic_transform = (
            mandibular_mic_to_rc_transform
            .inverse()
        )

        return CentricRelationRegistrationResult(
            record_to_mount_transform=(
                maxillary_registration.transform
            ),
            mandibular_mic_to_rc_transform=(
                mandibular_mic_to_rc_transform
            ),
            mandibular_rc_to_mic_transform=(
                mandibular_rc_to_mic_transform
            ),
            maxillary_registration=(
                maxillary_registration
            ),
            mandibular_registration=(
                mandibular_registration
            ),
        )