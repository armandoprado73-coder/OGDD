"""
OGDD - Protrusion Controller

Controls guided bilateral mandibular protrusion.
"""

from dataclasses import dataclass, field
import math

from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.articulator.guided_protrusion import (
    GuidedProtrusion,
    MandibularProtrusivePosition,
)


@dataclass
class ProtrusionController:
    """
    Stateful control for guided protrusive movement.

    Distance is expressed in millimeters. Zero represents
    centric relation. Positive values advance the mandible.

    Every position is calculated from the original
    mandibular assembly.
    """

    assembly: MandibularAssembly

    protrusion: GuidedProtrusion

    maximum_distance_mm: float

    step_mm: float = 1.0

    _distance_mm: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate the controller configuration.
        """

        self.maximum_distance_mm = float(
            self.maximum_distance_mm
        )

        self.step_mm = float(
            self.step_mm
        )

        if (
            not math.isfinite(
                self.maximum_distance_mm
            )
            or self.maximum_distance_mm <= 0.0
        ):
            raise ValueError(
                "Maximum protrusion distance must be "
                "positive and finite."
            )

        if (
            self.maximum_distance_mm
            > self.protrusion.maximum_translation
        ):
            raise ValueError(
                "Maximum protrusion distance cannot "
                "exceed the common guide limit."
            )

        if (
            not math.isfinite(self.step_mm)
            or self.step_mm <= 0.0
        ):
            raise ValueError(
                "Protrusion step must be positive "
                "and finite."
            )

        if self.step_mm > self.maximum_distance_mm:
            raise ValueError(
                "Protrusion step cannot exceed "
                "the maximum protrusion distance."
            )

    @property
    def distance_mm(self) -> float:
        """
        Current protrusive distance in millimeters.
        """

        return self._distance_mm

    @property
    def position(
        self,
    ) -> MandibularProtrusivePosition:
        """
        Mandibular position at the current distance.
        """

        return self.protrusion.position_at(
            assembly=self.assembly,
            distance=self._distance_mm,
        )

    @property
    def is_centered(self) -> bool:
        """
        Whether the controller is at centric relation.
        """

        return math.isclose(
            self._distance_mm,
            0.0,
            abs_tol=1e-12,
        )

    @property
    def is_at_limit(self) -> bool:
        """
        Whether maximum protrusion was reached.
        """

        return math.isclose(
            self._distance_mm,
            self.maximum_distance_mm,
            abs_tol=1e-12,
        )

    def set_distance(
        self,
        distance_mm: float,
    ) -> MandibularProtrusivePosition:
        """
        Move to an exact protrusive distance.
        """

        distance_mm = float(
            distance_mm
        )

        if not math.isfinite(distance_mm):
            raise ValueError(
                "Protrusion distance must be finite."
            )

        if not (
            0.0
            <= distance_mm
            <= self.maximum_distance_mm
        ):
            raise ValueError(
                "Protrusion distance must remain "
                "within the configured limits."
            )

        if math.isclose(
            distance_mm,
            0.0,
            abs_tol=1e-12,
        ):
            distance_mm = 0.0

        self._distance_mm = distance_mm

        return self.position

    def advance(
        self,
    ) -> MandibularProtrusivePosition:
        """
        Advance one configured step.
        """

        self._distance_mm = min(
            self._distance_mm + self.step_mm,
            self.maximum_distance_mm,
        )

        return self.position

    def retreat(
        self,
    ) -> MandibularProtrusivePosition:
        """
        Retreat one configured step toward centric.
        """

        self._distance_mm = max(
            self._distance_mm - self.step_mm,
            0.0,
        )

        return self.position

    def reset(
        self,
    ) -> MandibularProtrusivePosition:
        """
        Return to centric relation.
        """

        self._distance_mm = 0.0

        return self.position