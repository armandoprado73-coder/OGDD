"""
OGDD - Condylar Guide Builder

Builds the right and left condylar guides from
an anatomical hinge axis and an articulator
configuration.
"""

from dataclasses import dataclass

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.geometry.coordinate_system import (
    CoordinateSystem,
)

from .condylar_guide import CondylarGuide
from .configuration import ArticulatorConfiguration


@dataclass(frozen=True)
class CondylarGuidePair:
    """
    Anatomically identified pair of condylar guides.
    """

    right_guide: CondylarGuide
    left_guide: CondylarGuide


class CondylarGuideBuilder:
    """
    Builds independent right and left condylar guides.
    """

    @staticmethod
    def build(
        hinge_axis: HingeAxis,
        coordinate_system: CoordinateSystem,
        configuration: ArticulatorConfiguration,
    ) -> CondylarGuidePair:
        """
        Builds both guides from their condylar centers.

        Each guide receives its own sagittal guidance
        angle while sharing the anatomical coordinate
        system and mechanical dimensions.
        """

        right_guide = CondylarGuide(
            condyle_center=(
                hinge_axis.right_condyle.point
            ),
            coordinate_system=coordinate_system,
            angle_degrees=(
                configuration
                .right_condylar_guidance_degrees
            ),
            condyle_diameter=(
                configuration.condyle_diameter
            ),
            length=(
                configuration.condylar_guide_length
            ),
            width=(
                configuration.condylar_guide_width
            ),
        )

        left_guide = CondylarGuide(
            condyle_center=(
                hinge_axis.left_condyle.point
            ),
            coordinate_system=coordinate_system,
            angle_degrees=(
                configuration
                .left_condylar_guidance_degrees
            ),
            condyle_diameter=(
                configuration.condyle_diameter
            ),
            length=(
                configuration.condylar_guide_length
            ),
            width=(
                configuration.condylar_guide_width
            ),
        )

        return CondylarGuidePair(
            right_guide=right_guide,
            left_guide=left_guide,
        )