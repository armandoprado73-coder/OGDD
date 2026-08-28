"""
OGDD - Lateral Excursion

Mandibular lateral movement around the working condyle.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.geometry.transform import Transform


class LateralSide(Enum):
    """
    Working side of a mandibular lateral excursion.
    """

    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class LateralExcursion:
    """
    Mandibular lateral excursion around the working condyle.

    The working condyle remains fixed and acts as the
    center of rotation. The balancing condyle translates
    along an arc while preserving the intercondylar
    distance.

    The excursion angle represents mandibular rotation,
    not the Bennett angle.
    """

    hinge_axis: HingeAxis

    superior_direction: np.ndarray

    working_side: LateralSide

    def __post_init__(self) -> None:
        """
        Validate and normalize the superior direction.
        """

        superior_direction = np.asarray(
            self.superior_direction,
            dtype=float,
        )

        if superior_direction.shape != (3,):
            raise ValueError(
                "The superior direction must be a 3D vector."
            )

        direction_length = np.linalg.norm(
            superior_direction
        )

        if np.isclose(direction_length, 0.0):
            raise ValueError(
                "The superior direction cannot have zero length."
            )

        if not isinstance(
            self.working_side,
            LateralSide,
        ):
            raise ValueError(
                "The working side must be a LateralSide."
            )

        object.__setattr__(
            self,
            "superior_direction",
            superior_direction / direction_length,
        )

    @property
    def working_condyle(self) -> Landmark:
        """
        Condyle that remains fixed and rotates.
        """

        if self.working_side is LateralSide.RIGHT:
            return self.hinge_axis.right_condyle

        return self.hinge_axis.left_condyle

    @property
    def balancing_condyle(self) -> Landmark:
        """
        Condyle that translates during lateral movement.
        """

        if self.working_side is LateralSide.RIGHT:
            return self.hinge_axis.left_condyle

        return self.hinge_axis.right_condyle

    def transform_at(
        self,
        angle_degrees: float,
    ) -> Transform:
        """
        Build the rigid transformation for an excursion.

        Positive input angles move the balancing condyle
        toward the anterior region on either working side.
        """

        angle_degrees = float(
            angle_degrees
        )

        if angle_degrees < 0.0:
            raise ValueError(
                "The excursion angle cannot be negative."
            )

        if self.working_side is LateralSide.RIGHT:
            signed_angle = -angle_degrees
        else:
            signed_angle = angle_degrees

        return Transform.rotation_about_axis(
            origin=self.working_condyle.point,
            axis=self.superior_direction,
            angle_degrees=signed_angle,
        )

    def rotate_points(
        self,
        points: np.ndarray,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Rotate 3D points around the working condyle.
        """

        return self.transform_at(
            angle_degrees=angle_degrees,
        ).apply(points)

    def working_condyle_at(
        self,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Return the fixed working-condyle center.
        """

        result = self.rotate_points(
            points=np.array(
                [
                    self.working_condyle.point,
                ]
            ),
            angle_degrees=angle_degrees,
        )

        return result[0]

    def balancing_condyle_at(
        self,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Return the translated balancing-condyle center.
        """

        result = self.rotate_points(
            points=np.array(
                [
                    self.balancing_condyle.point,
                ]
            ),
            angle_degrees=angle_degrees,
        )

        return result[0]