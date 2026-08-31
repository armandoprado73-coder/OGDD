"""
OGDD - Combined Mandibular Movement

Composes guided lateral excursion, protrusion and
opening into one rigid mandibular transformation.
"""

from dataclasses import dataclass
import math

import numpy as np

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.articulator.guided_lateral_excursion import (
    GuidedLateralExcursion,
)
from ogdd.articulator.guided_protrusion import (
    GuidedProtrusion,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)
from ogdd.geometry.transform import Transform
from ogdd.mesh import Mesh


@dataclass(frozen=True)
class MandibularCombinedPosition:
    """
    Complete position produced by combined movement.
    """

    mesh: Mesh

    balkwill: BalkwillTriangle

    bonwill: BonwillTriangle

    hinge_axis: HingeAxis

    opening_angle_degrees: float

    lateral_angle_degrees: float

    protrusion_distance_mm: float

    working_side: LateralSide | None


@dataclass(frozen=True)
class CombinedMovement:
    """
    Combine the three current mandibular movements.

    Lateral excursion is evaluated first from centric
    relation. Protrusion then translates that complete
    lateral position. Opening is finally performed around
    the resulting mobile intercondylar axis.

    Every position is calculated from the original
    assembly, so transformations never accumulate.
    """

    assembly: MandibularAssembly

    right_excursion: GuidedLateralExcursion

    left_excursion: GuidedLateralExcursion

    protrusion: GuidedProtrusion

    def __post_init__(self) -> None:
        """
        Validate that every movement shares one assembly.
        """

        if (
            self.right_excursion.working_side
            is not LateralSide.RIGHT
        ):
            raise ValueError(
                "The right excursion must use "
                "the right working side."
            )

        if (
            self.left_excursion.working_side
            is not LateralSide.LEFT
        ):
            raise ValueError(
                "The left excursion must use "
                "the left working side."
            )

        self._validate_shared_hinge_axis(
            hinge_axis=(
                self.right_excursion.hinge_axis
            ),
            movement_name="right excursion",
        )

        self._validate_shared_hinge_axis(
            hinge_axis=(
                self.left_excursion.hinge_axis
            ),
            movement_name="left excursion",
        )

        self._validate_shared_hinge_axis(
            hinge_axis=self.protrusion.hinge_axis,
            movement_name="protrusion",
        )

    def _validate_shared_hinge_axis(
        self,
        hinge_axis: HingeAxis,
        movement_name: str,
    ) -> None:
        """
        Require a movement to use the assembly condyles.
        """

        if not np.allclose(
            hinge_axis.left_condyle.point,
            self.assembly.hinge_axis.left_condyle.point,
        ):
            raise ValueError(
                f"The {movement_name} and assembly must "
                "share the same left condyle."
            )

        if not np.allclose(
            hinge_axis.right_condyle.point,
            self.assembly.hinge_axis.right_condyle.point,
        ):
            raise ValueError(
                f"The {movement_name} and assembly must "
                "share the same right condyle."
            )

    @staticmethod
    def _validated_opening_angle(
        opening_angle_degrees: float,
    ) -> float:
        """
        Validate a non-negative opening angle.
        """

        opening_angle_degrees = float(
            opening_angle_degrees
        )

        if (
            not math.isfinite(opening_angle_degrees)
            or opening_angle_degrees < 0.0
        ):
            raise ValueError(
                "The opening angle must be "
                "non-negative and finite."
            )

        return opening_angle_degrees

    def _lateral_components(
        self,
        lateral_angle_degrees: float,
    ) -> tuple[
        GuidedLateralExcursion | None,
        float,
        LateralSide | None,
    ]:
        """
        Resolve a signed lateral angle and working side.
        """

        lateral_angle_degrees = float(
            lateral_angle_degrees
        )

        if not math.isfinite(
            lateral_angle_degrees
        ):
            raise ValueError(
                "The lateral angle must be finite."
            )

        if math.isclose(
            lateral_angle_degrees,
            0.0,
            abs_tol=1e-12,
        ):
            return None, 0.0, None

        if lateral_angle_degrees > 0.0:
            excursion = self.right_excursion
            angle_magnitude = lateral_angle_degrees
            working_side = LateralSide.RIGHT
        else:
            excursion = self.left_excursion
            angle_magnitude = -lateral_angle_degrees
            working_side = LateralSide.LEFT

        if (
            angle_magnitude
            > excursion.maximum_angle_degrees
        ):
            raise ValueError(
                "The lateral angle exceeds the "
                "condylar guide limit."
            )

        return (
            excursion,
            angle_magnitude,
            working_side,
        )

    def _validate_combined_guide_travel(
        self,
        excursion: GuidedLateralExcursion | None,
        lateral_angle_magnitude: float,
        protrusion_distance_mm: float,
    ) -> None:
        """
        Protect the shared balancing-guide trajectory.

        Protrusion advances both condyles. Lateral
        excursion adds more travel on the balancing side,
        so their sum must remain inside that guide.
        """

        if excursion is None:
            return

        balancing_travel = (
            protrusion_distance_mm
            + excursion.guide_distance_at(
                lateral_angle_magnitude
            )
        )

        if (
            balancing_travel
            > excursion
            .balancing_guide
            .maximum_translation
            + 1e-12
        ):
            raise ValueError(
                "The combined movement exceeds the "
                "balancing condylar guide limit."
            )

    def preopening_transform_at(
        self,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
    ) -> Transform:
        """
        Compose lateral excursion and protrusion.
        """

        (
            excursion,
            angle_magnitude,
            _,
        ) = self._lateral_components(
            lateral_angle_degrees
        )

        protrusion_distance_mm = (
            self.protrusion.validated_distance(
                protrusion_distance_mm
            )
        )

        self._validate_combined_guide_travel(
            excursion=excursion,
            lateral_angle_magnitude=angle_magnitude,
            protrusion_distance_mm=(
                protrusion_distance_mm
            ),
        )

        lateral_transform = Transform.identity()

        if excursion is not None:
            lateral_transform = excursion.transform_at(
                angle_magnitude
            )

        protrusion_transform = (
            self.protrusion.transform_at(
                protrusion_distance_mm
            )
        )

        return Transform(
            protrusion_transform.matrix
            @ lateral_transform.matrix
        )

    def mobile_hinge_axis_at(
        self,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
    ) -> HingeAxis:
        """
        Return the axis after condylar translation.
        """

        transform = self.preopening_transform_at(
            lateral_angle_degrees=(
                lateral_angle_degrees
            ),
            protrusion_distance_mm=(
                protrusion_distance_mm
            ),
        )

        left_condyle = self._transform_landmark(
            landmark=(
                self.assembly.bonwill.left_condyle
            ),
            transform=transform,
        )

        right_condyle = self._transform_landmark(
            landmark=(
                self.assembly.bonwill.right_condyle
            ),
            transform=transform,
        )

        return HingeAxis(
            left_condyle=left_condyle,
            right_condyle=right_condyle,
        )

    def transform_at(
        self,
        opening_angle_degrees: float,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
    ) -> Transform:
        """
        Build the complete rigid transformation.
        """

        opening_angle_degrees = (
            self._validated_opening_angle(
                opening_angle_degrees
            )
        )

        preopening_transform = (
            self.preopening_transform_at(
                lateral_angle_degrees=(
                    lateral_angle_degrees
                ),
                protrusion_distance_mm=(
                    protrusion_distance_mm
                ),
            )
        )

        if math.isclose(
            opening_angle_degrees,
            0.0,
            abs_tol=1e-12,
        ):
            return preopening_transform

        mobile_hinge_axis = self.mobile_hinge_axis_at(
            lateral_angle_degrees=(
                lateral_angle_degrees
            ),
            protrusion_distance_mm=(
                protrusion_distance_mm
            ),
        )

        opening_transform = (
            Transform.rotation_about_axis(
                origin=mobile_hinge_axis.midpoint,
                axis=mobile_hinge_axis.direction,
                angle_degrees=(
                    -opening_angle_degrees
                ),
            )
        )

        return Transform(
            opening_transform.matrix
            @ preopening_transform.matrix
        )

    @staticmethod
    def _transform_landmark(
        landmark: Landmark,
        transform: Transform,
    ) -> Landmark:
        """
        Transform one landmark and preserve metadata.
        """

        transformed_point = transform.apply(
            np.array([
                landmark.point,
            ])
        )[0]

        return Landmark(
            name=landmark.name,
            point=transformed_point,
            reference_used=landmark.reference_used,
            confidence=landmark.confidence,
            created_by=landmark.created_by,
        )

    def _transform_balkwill(
        self,
        transform: Transform,
    ) -> BalkwillTriangle:
        """
        Transform the complete Balkwill triangle.
        """

        balkwill = self.assembly.balkwill

        return BalkwillTriangle(
            left_posterior=self._transform_landmark(
                balkwill.left_posterior,
                transform,
            ),
            right_posterior=self._transform_landmark(
                balkwill.right_posterior,
                transform,
            ),
            dental_midline=self._transform_landmark(
                balkwill.dental_midline,
                transform,
            ),
        )

    def _transform_bonwill(
        self,
        transform: Transform,
    ) -> BonwillTriangle:
        """
        Transform the complete Bonwill triangle.
        """

        bonwill = self.assembly.bonwill

        return BonwillTriangle(
            left_condyle=self._transform_landmark(
                bonwill.left_condyle,
                transform,
            ),
            right_condyle=self._transform_landmark(
                bonwill.right_condyle,
                transform,
            ),
            dental_midline=self._transform_landmark(
                bonwill.dental_midline,
                transform,
            ),
        )

    def _transform_mesh(
        self,
        transform: Transform,
    ) -> Mesh:
        """
        Transform the mandibular mesh rigidly.
        """

        mesh = self.assembly.mesh

        transformed_normals = None

        if mesh.normals is not None:
            transformed_normals = (
                transform.apply_vectors(
                    mesh.normals
                )
            )

        return Mesh(
            vertices=transform.apply(mesh.vertices),
            faces=mesh.faces.copy(),
            normals=transformed_normals,
            attributes={
                name: values.copy()
                for name, values
                in mesh.attributes.items()
            },
            metadata=mesh.metadata.copy(),
        )

    def position_at(
        self,
        opening_angle_degrees: float,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
    ) -> MandibularCombinedPosition:
        """
        Calculate one complete combined position.
        """

        transform = self.transform_at(
            opening_angle_degrees=(
                opening_angle_degrees
            ),
            lateral_angle_degrees=(
                lateral_angle_degrees
            ),
            protrusion_distance_mm=(
                protrusion_distance_mm
            ),
        )

        opening_angle_degrees = float(
            opening_angle_degrees
        )
        lateral_angle_degrees = float(
            lateral_angle_degrees
        )
        protrusion_distance_mm = float(
            protrusion_distance_mm
        )

        transformed_bonwill = (
            self._transform_bonwill(transform)
        )

        transformed_hinge_axis = HingeAxis(
            left_condyle=(
                transformed_bonwill.left_condyle
            ),
            right_condyle=(
                transformed_bonwill.right_condyle
            ),
        )

        _, _, working_side = (
            self._lateral_components(
                lateral_angle_degrees
            )
        )

        return MandibularCombinedPosition(
            mesh=self._transform_mesh(transform),
            balkwill=self._transform_balkwill(
                transform
            ),
            bonwill=transformed_bonwill,
            hinge_axis=transformed_hinge_axis,
            opening_angle_degrees=(
                opening_angle_degrees
            ),
            lateral_angle_degrees=(
                lateral_angle_degrees
            ),
            protrusion_distance_mm=(
                protrusion_distance_mm
            ),
            working_side=working_side,
        )
