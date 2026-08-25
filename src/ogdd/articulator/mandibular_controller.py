"""
OGDD - Mandibular Controller

Controls incremental opening and closing of a
mandibular assembly.
"""

from dataclasses import dataclass, field
import math

from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
    MandibularPosition,
)


@dataclass
class MandibularController:
    """
    Stateful control for mandibular hinge movement.

    The controller stores the current opening angle.
    Every position is still calculated by the assembly
    from its original closed state.
    """

    assembly: MandibularAssembly

    maximum_angle_degrees: float

    step_degrees: float = 1.0

    _angle_degrees: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validates the movement configuration.
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
                "Maximum opening angle must be "
                "positive and finite."
            )

        if (
            not math.isfinite(self.step_degrees)
            or self.step_degrees <= 0.0
        ):
            raise ValueError(
                "Opening step must be positive "
                "and finite."
            )

        if (
            self.step_degrees
            > self.maximum_angle_degrees
        ):
            raise ValueError(
                "Opening step cannot exceed "
                "the maximum opening angle."
            )

    @property
    def angle_degrees(self) -> float:
        """
        Current mandibular opening angle.
        """

        return self._angle_degrees

    @property
    def position(self) -> MandibularPosition:
        """
        Mandibular position at the current angle.
        """

        return self.assembly.position_at(
            angle_degrees=self._angle_degrees,
        )

    @property
    def is_closed(self) -> bool:
        """
        Whether the controller is at zero degrees.
        """

        return math.isclose(
            self._angle_degrees,
            0.0,
            abs_tol=1e-12,
        )

    @property
    def is_fully_open(self) -> bool:
        """
        Whether the configured maximum was reached.
        """

        return math.isclose(
            self._angle_degrees,
            self.maximum_angle_degrees,
            abs_tol=1e-12,
        )

    def set_angle(
        self,
        angle_degrees: float,
    ) -> MandibularPosition:
        """
        Moves the mandible to an exact opening angle.
        """

        angle_degrees = float(angle_degrees)

        if not math.isfinite(angle_degrees):
            raise ValueError(
                "Opening angle must be finite."
            )

        if not (
            0.0
            <= angle_degrees
            <= self.maximum_angle_degrees
        ):
            raise ValueError(
                "Opening angle must be between "
                "zero and the configured maximum."
            )

        self._angle_degrees = angle_degrees

        return self.position

    def open(self) -> MandibularPosition:
        """
        Opens the mandible by one configured step.
        """

        self._angle_degrees = min(
            self._angle_degrees + self.step_degrees,
            self.maximum_angle_degrees,
        )

        return self.position

    def close(self) -> MandibularPosition:
        """
        Closes the mandible by one configured step.
        """

        self._angle_degrees = max(
            self._angle_degrees - self.step_degrees,
            0.0,
        )

        return self.position

    def reset(self) -> MandibularPosition:
        """
        Returns the mandible to the closed position.
        """

        self._angle_degrees = 0.0

        return self.position