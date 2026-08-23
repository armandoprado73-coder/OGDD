"""
OGDD - Mandibular Assembly

Groups the mandibular mesh and its anatomical
references into a single articulated structure.
"""

from dataclasses import dataclass

import numpy as np

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.mesh import Mesh


@dataclass(frozen=True)
class MandibularPosition:
    """
    Mandibular assembly evaluated at a specific
    opening angle.
    """

    mesh: Mesh

    balkwill: BalkwillTriangle

    bonwill: BonwillTriangle

    hinge_axis: HingeAxis

    angle_degrees: float


@dataclass(frozen=True)
class MandibularAssembly:
    """
    Complete mandibular articulated assembly.

    The assembly stores the original closed position.
    New positions are always calculated from this
    original state to avoid accumulated errors.
    """

    mesh: Mesh

    balkwill: BalkwillTriangle

    bonwill: BonwillTriangle

    hinge_axis: HingeAxis

    def __post_init__(self) -> None:
        """
        Validates that all anatomical structures
        belong to the same mandibular assembly.
        """

        if not np.allclose(
            self.balkwill.dental_midline.point,
            self.bonwill.dental_midline.point,
        ):
            raise ValueError(
                "Balkwill and Bonwill must share "
                "the same dental midline."
            )

        if not np.allclose(
            self.hinge_axis.left_condyle.point,
            self.bonwill.left_condyle.point,
        ):
            raise ValueError(
                "The left condyle must match "
                "the hinge axis."
            )

        if not np.allclose(
            self.hinge_axis.right_condyle.point,
            self.bonwill.right_condyle.point,
        ):
            raise ValueError(
                "The right condyle must match "
                "the hinge axis."
            )

    def position_at(
        self,
        angle_degrees: float,
    ) -> MandibularPosition:
        """
        Calculates the mandibular position at an
        opening angle.

        Every position is calculated from the original
        closed assembly.
        """

        angle_degrees = float(angle_degrees)

        if angle_degrees < 0.0:
            raise ValueError(
                "The opening angle cannot be negative."
            )

        rotated_mesh = self.hinge_axis.rotate_mesh(
            mesh=self.mesh,
            angle_degrees=angle_degrees,
        )

        rotated_balkwill = (
            self.hinge_axis.rotate_balkwill(
                balkwill=self.balkwill,
                angle_degrees=angle_degrees,
            )
        )

        rotated_bonwill = (
            self.hinge_axis.rotate_bonwill(
                bonwill=self.bonwill,
                angle_degrees=angle_degrees,
            )
        )

        return MandibularPosition(
            mesh=rotated_mesh,
            balkwill=rotated_balkwill,
            bonwill=rotated_bonwill,
            hinge_axis=self.hinge_axis,
            angle_degrees=angle_degrees,
        )