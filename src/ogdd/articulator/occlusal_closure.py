"""
OGDD - Occlusal Closure

Applies operator-controlled opening or closure around
the mobile hinge axis of a combined mandibular position.
"""

from dataclasses import dataclass
import math

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.articulator.combined_movement import (
    MandibularCombinedPosition,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)
from ogdd.mesh import Mesh


@dataclass(frozen=True)
class MandibularOcclusalPosition:
    """
    Complete position after manual occlusal adjustment.

    Positive adjustment angles add opening. Negative
    adjustment angles add closure around the mobile axis.
    """

    mesh: Mesh

    balkwill: BalkwillTriangle

    bonwill: BonwillTriangle

    hinge_axis: HingeAxis

    base_opening_angle_degrees: float

    adjustment_angle_degrees: float

    total_opening_angle_degrees: float

    lateral_angle_degrees: float

    protrusion_distance_mm: float

    working_side: LateralSide | None


@dataclass(frozen=True)
class OcclusalClosure:
    """
    Apply a signed adjustment around a mobile hinge axis.

    The supplied combined position is the immutable base.
    Every adjustment is calculated directly from that base
    to avoid accumulated rotational error.

    The primitive intentionally does not choose a clinical
    endpoint. The operator remains responsible for finding
    and confirming the desired dental relationship.
    """

    def validated_adjustment(
        self,
        adjustment_angle_degrees: float,
    ) -> float:
        """
        Validate a finite signed adjustment angle.
        """

        adjustment_angle_degrees = float(
            adjustment_angle_degrees
        )

        if not math.isfinite(
            adjustment_angle_degrees
        ):
            raise ValueError(
                "Occlusal adjustment angle must be finite."
            )

        if math.isclose(
            adjustment_angle_degrees,
            0.0,
            abs_tol=1e-12,
        ):
            return 0.0

        return adjustment_angle_degrees

    def position_at(
        self,
        position: MandibularCombinedPosition,
        adjustment_angle_degrees: float,
    ) -> MandibularOcclusalPosition:
        """
        Return an operator-adjusted mandibular position.

        Positive values open and negative values close.
        Both condyles remain fixed because adjustment occurs
        around the already repositioned intercondylar axis.
        """

        adjustment_angle_degrees = (
            self.validated_adjustment(
                adjustment_angle_degrees
            )
        )

        adjusted_mesh = position.hinge_axis.rotate_mesh(
            mesh=position.mesh,
            angle_degrees=adjustment_angle_degrees,
        )

        adjusted_balkwill = (
            position.hinge_axis.rotate_balkwill(
                balkwill=position.balkwill,
                angle_degrees=(
                    adjustment_angle_degrees
                ),
            )
        )

        adjusted_bonwill = (
            position.hinge_axis.rotate_bonwill(
                bonwill=position.bonwill,
                angle_degrees=(
                    adjustment_angle_degrees
                ),
            )
        )

        adjusted_hinge_axis = HingeAxis(
            left_condyle=adjusted_bonwill.left_condyle,
            right_condyle=adjusted_bonwill.right_condyle,
        )

        return MandibularOcclusalPosition(
            mesh=adjusted_mesh,
            balkwill=adjusted_balkwill,
            bonwill=adjusted_bonwill,
            hinge_axis=adjusted_hinge_axis,
            base_opening_angle_degrees=(
                position.opening_angle_degrees
            ),
            adjustment_angle_degrees=(
                adjustment_angle_degrees
            ),
            total_opening_angle_degrees=(
                position.opening_angle_degrees
                + adjustment_angle_degrees
            ),
            lateral_angle_degrees=(
                position.lateral_angle_degrees
            ),
            protrusion_distance_mm=(
                position.protrusion_distance_mm
            ),
            working_side=position.working_side,
        )
