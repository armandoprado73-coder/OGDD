"""
OGDD - Lateral Controller

Controls right and left mandibular excursions.
"""

from dataclasses import dataclass, field
import math

from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.articulator.lateral_excursion import (
    LateralExcursion,
    LateralSide,
    MandibularLateralPosition,
)


@dataclass
class LateralController:
    """
    Stateful control for bilateral mandibular movement.

    Negative angles represent left lateral excursion.
    Positive angles represent right lateral excursion.
    Zero represents centric relation.

    Every position is calculated from the original
    mandibular assembly.
    """

    assembly: MandibularAssembly

    right_excursion: LateralExcursion

    left_excursion: LateralExcursion

    maximum_angle_degrees: float

    step_degrees: float = 1.0

    _angle_degrees: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate the controller configuration.
        """

        self.maximum_angle_degrees = float(
            self.maximum_angle_degrees
        )

        self.step_degrees = float(
            self.step_degrees
        )

        if (
            not math.isfinite(
                self.maximum_angle_degrees
            )
            or self.maximum_angle_degrees <= 0.0
        ):
            raise ValueError(
                "Maximum lateral angle must be "
                "positive and finite."
            )

        if (
            not math.isfinite(self.step_degrees)
            or self.step_degrees <= 0.0
        ):
            raise ValueError(
                "Lateral step must be positive "
                "and finite."
            )

        if (
            self.step_degrees
            > self.maximum_angle_degrees
        ):
            raise ValueError(
                "Lateral step cannot exceed "
                "the maximum lateral angle."
            )

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

    @property
    def angle_degrees(self) -> float:
        """
        Current signed lateral angle.
        """

        return self._angle_degrees

    @property
    def working_side(self) -> LateralSide | None:
        """
        Current working side, or None at centric relation.
        """

        if self.is_centered:
            return None

        if self._angle_degrees > 0.0:
            return LateralSide.RIGHT

        return LateralSide.LEFT

    @property
    def position(self) -> MandibularLateralPosition:
        """
        Mandibular position at the current lateral angle.

        At zero, the right excursion is used only as a
        computational convention. The resulting geometry
        is the original centric position.
        """

        if self._angle_degrees >= 0.0:
            return self.right_excursion.position_at(
                assembly=self.assembly,
                angle_degrees=self._angle_degrees,
            )

        return self.left_excursion.position_at(
            assembly=self.assembly,
            angle_degrees=-self._angle_degrees,
        )

    @property
    def is_centered(self) -> bool:
        """
        Whether the controller is at centric relation.
        """

        return math.isclose(
            self._angle_degrees,
            0.0,
            abs_tol=1e-12,
        )

    @property
    def is_at_right_limit(self) -> bool:
        """
        Whether maximum right excursion was reached.
        """

        return math.isclose(
            self._angle_degrees,
            self.maximum_angle_degrees,
            abs_tol=1e-12,
        )

    @property
    def is_at_left_limit(self) -> bool:
        """
        Whether maximum left excursion was reached.
        """

        return math.isclose(
            self._angle_degrees,
            -self.maximum_angle_degrees,
            abs_tol=1e-12,
        )

    def set_angle(
        self,
        angle_degrees: float,
    ) -> MandibularLateralPosition:
        """
        Move to an exact signed lateral angle.
        """

        angle_degrees = float(
            angle_degrees
        )

        if not math.isfinite(angle_degrees):
            raise ValueError(
                "Lateral angle must be finite."
            )

        if not (
            -self.maximum_angle_degrees
            <= angle_degrees
            <= self.maximum_angle_degrees
        ):
            raise ValueError(
                "Lateral angle must remain within "
                "the configured limits."
            )

        if math.isclose(
            angle_degrees,
            0.0,
            abs_tol=1e-12,
        ):
            angle_degrees = 0.0

        self._angle_degrees = angle_degrees

        return self.position

    def move_right(
        self,
    ) -> MandibularLateralPosition:
        """
        Move one configured step toward the right.
        """

        self._angle_degrees = min(
            self._angle_degrees + self.step_degrees,
            self.maximum_angle_degrees,
        )

        return self.position

    def move_left(
        self,
    ) -> MandibularLateralPosition:
        """
        Move one configured step toward the left.
        """

        self._angle_degrees = max(
            self._angle_degrees - self.step_degrees,
            -self.maximum_angle_degrees,
        )

        return self.position

    def reset(
        self,
    ) -> MandibularLateralPosition:
        """
        Return to centric relation.
        """

        self._angle_degrees = 0.0

        return self.position