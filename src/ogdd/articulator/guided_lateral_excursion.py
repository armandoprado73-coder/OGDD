"""
OGDD - Guided Lateral Excursion

Three-dimensional lateral movement constrained by
the balancing condylar guide.
"""

from dataclasses import dataclass
import math

import numpy as np

from ogdd.articulator.condylar_guide import (
    CondylarGuide,
)
from ogdd.articulator.lateral_excursion import (
    LateralExcursion,
)
from ogdd.geometry.transform import Transform


@dataclass(frozen=True)
class GuidedLateralExcursion(
    LateralExcursion
):
    """
    Lateral excursion constrained by a condylar guide.

    The working condyle remains fixed. The balancing
    condyle advances, descends and moves medially while
    preserving the original intercondylar distance.
    """

    balancing_guide: CondylarGuide

    def __post_init__(self) -> None:
        """
        Validate the excursion and balancing guide.
        """

        super().__post_init__()

        if not np.allclose(
            self.balancing_guide.condyle_center,
            self.balancing_condyle.point,
        ):
            raise ValueError(
                "The guide must belong to the "
                "balancing condyle."
            )

        trajectory_axis_dot = np.dot(
            self.balancing_guide.trajectory_direction,
            self.hinge_axis.direction,
        )

        if not np.isclose(
            trajectory_axis_dot,
            0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "The guide trajectory must be "
                "orthogonal to the hinge axis."
            )

        if (
            self.balancing_guide.maximum_translation
            > self.hinge_axis.length
        ):
            raise ValueError(
                "The guide translation cannot exceed "
                "the intercondylar distance."
            )

    @property
    def maximum_angle_degrees(self) -> float:
        """
        Maximum rotation permitted by the guide length.
        """

        ratio = (
            self.balancing_guide.maximum_translation
            / self.hinge_axis.length
        )

        return math.degrees(
            math.asin(ratio)
        )

    def guide_distance_at(
        self,
        angle_degrees: float,
    ) -> float:
        """
        Convert mandibular rotation into guide travel.
        """

        angle_degrees = float(
            angle_degrees
        )

        if (
            not math.isfinite(angle_degrees)
            or angle_degrees < 0.0
        ):
            raise ValueError(
                "The excursion angle must be "
                "non-negative and finite."
            )

        if (
            angle_degrees
            > self.maximum_angle_degrees
        ):
            raise ValueError(
                "The excursion angle exceeds "
                "the condylar guide limit."
            )

        angle_radians = math.radians(
            angle_degrees
        )

        return (
            self.hinge_axis.length
            * math.sin(angle_radians)
        )

    def balancing_target_at(
        self,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Return the guided balancing-condyle position.

        Guide travel provides the anterior-inferior
        component. A medial correction preserves the
        intercondylar distance.
        """

        guide_distance = self.guide_distance_at(
            angle_degrees
        )

        guided_center = (
            self.balancing_guide.center_at(
                guide_distance
            )
        )

        intercondylar_width = (
            self.hinge_axis.length
        )

        remaining_transverse_distance = math.sqrt(
            max(
                intercondylar_width ** 2
                - guide_distance ** 2,
                0.0,
            )
        )

        medial_correction = (
            intercondylar_width
            - remaining_transverse_distance
        )

        direction_toward_working = (
            self.working_condyle.point
            - self.balancing_condyle.point
        ) / intercondylar_width

        return (
            guided_center
            + medial_correction
            * direction_toward_working
        )

    def transform_at(
        self,
        angle_degrees: float,
    ) -> Transform:
        """
        Build the rigid guided lateral transformation.
        """

        angle_degrees = float(
            angle_degrees
        )

        if math.isclose(
            angle_degrees,
            0.0,
            abs_tol=1e-12,
        ):
            return Transform.identity()

        target = self.balancing_target_at(
            angle_degrees
        )

        initial_vector = (
            self.balancing_condyle.point
            - self.working_condyle.point
        )

        target_vector = (
            target
            - self.working_condyle.point
        )

        rotation_axis = np.cross(
            initial_vector,
            target_vector,
        )

        rotation_axis_length = np.linalg.norm(
            rotation_axis
        )

        if np.isclose(
            rotation_axis_length,
            0.0,
        ):
            raise ValueError(
                "The guided movement cannot define "
                "a rotation axis."
            )

        cosine = np.dot(
            initial_vector,
            target_vector,
        ) / (
            np.linalg.norm(initial_vector)
            * np.linalg.norm(target_vector)
        )

        cosine = float(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )

        rotation_degrees = math.degrees(
            math.acos(cosine)
        )

        return Transform.rotation_about_axis(
            origin=self.working_condyle.point,
            axis=rotation_axis,
            angle_degrees=rotation_degrees,
        )