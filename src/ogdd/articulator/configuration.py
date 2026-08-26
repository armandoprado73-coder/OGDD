"""
OGDD - Articulator Configuration

Configuration parameters for a virtual dental articulator.
"""

from dataclasses import dataclass
from enum import Enum
import math


class IntercondylarPreset(Enum):
    """
    Standard intercondylar width presets in millimeters.
    """

    S = 95.0
    M = 110.0
    L = 140.0


@dataclass(frozen=True)
class ArticulatorConfiguration:
    """
    Mechanical configuration of a virtual articulator.
    """

    intercondylar_width: float = IntercondylarPreset.M.value
    balkwill_angle_degrees: float = 25.0

    right_condylar_guidance_degrees: float = 45.0
    left_condylar_guidance_degrees: float = 45.0

    condyle_diameter: float = 6.0

    condylar_guide_length: float = 20.0
    condylar_guide_width: float = 20.0

    def __post_init__(self) -> None:
        """
        Validates the mechanical configuration.
        """

        if (
            not math.isfinite(self.intercondylar_width)
            or self.intercondylar_width <= 0.0
        ):
            raise ValueError(
                "Intercondylar width must be positive "
                "and finite."
            )

        if (
            not math.isfinite(self.balkwill_angle_degrees)
            or not 0.0
            < self.balkwill_angle_degrees
            < 90.0
        ):
            raise ValueError(
                "Balkwill angle must be between "
                "0 and 90 degrees."
            )

        for angle in (
            self.right_condylar_guidance_degrees,
            self.left_condylar_guidance_degrees,
        ):
            if (
                not math.isfinite(angle)
                or not 0.0 <= angle < 90.0
            ):
                raise ValueError(
                    "Condylar guidance angles must be "
                    "between 0 and 90 degrees."
                )

        if (
            not math.isfinite(self.condyle_diameter)
            or self.condyle_diameter <= 0.0
        ):
            raise ValueError(
                "Condyle diameter must be positive "
                "and finite."
            )

        if (
            not math.isfinite(self.condylar_guide_length)
            or self.condylar_guide_length <= 0.0
        ):
            raise ValueError(
                "Condylar guide length must be "
                "positive and finite."
            )

        if (
            not math.isfinite(self.condylar_guide_width)
            or self.condylar_guide_width <= 0.0
        ):
            raise ValueError(
                "Condylar guide width must be "
                "positive and finite."
            )

    @property
    def bonwill_side_length(self) -> float:
        """
        Side length of the virtual Bonwill triangle.
        """

        return self.intercondylar_width

    @property
    def condyle_radius(self) -> float:
        """
        Radius of each virtual condyle.
        """

        return self.condyle_diameter / 2.0

    @classmethod
    def from_preset(
        cls,
        preset: IntercondylarPreset,
    ) -> "ArticulatorConfiguration":
        """
        Creates an articulator configuration
        from a standard intercondylar preset.
        """

        return cls(
            intercondylar_width=preset.value,
        )