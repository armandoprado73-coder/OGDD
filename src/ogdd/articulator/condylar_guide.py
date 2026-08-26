"""
OGDD - Condylar Guide

Mechanical guide for a virtual articulator condyle.
"""

from dataclasses import dataclass
import math

import numpy as np

from ogdd.geometry.coordinate_system import CoordinateSystem


@dataclass(frozen=True)
class CondylarGuide:
    """
    Mechanical guide associated with one condyle.

    The guide contains a main inclined surface and
    a posterior stop. Both surfaces are tangent to
    the spherical condyle in centric position.

    Length represents the complete physical length
    of the main surface. Available translation is
    therefore:

        length - condyle radius
    """

    condyle_center: np.ndarray
    coordinate_system: CoordinateSystem

    angle_degrees: float = 45.0
    condyle_diameter: float = 6.0

    length: float = 20.0
    width: float = 20.0

    def __post_init__(self) -> None:
        """
        Validates and protects the guide geometry.
        """

        center = np.asarray(
            self.condyle_center,
            dtype=float,
        )

        if center.shape != (3,):
            raise ValueError(
                "Condyle center must be a 3D point."
            )

        if not np.all(np.isfinite(center)):
            raise ValueError(
                "Condyle center must contain "
                "finite values."
            )

        object.__setattr__(
            self,
            "condyle_center",
            center.copy(),
        )

        if (
            not math.isfinite(self.angle_degrees)
            or not 0.0 <= self.angle_degrees < 90.0
        ):
            raise ValueError(
                "Condylar guidance angle must be "
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
            not math.isfinite(self.length)
            or self.length <= 0.0
        ):
            raise ValueError(
                "Guide length must be positive "
                "and finite."
            )

        if self.length <= self.condyle_radius:
            raise ValueError(
                "Guide length must exceed "
                "the condyle radius."
            )

        if (
            not math.isfinite(self.width)
            or self.width <= 0.0
        ):
            raise ValueError(
                "Guide width must be positive "
                "and finite."
            )

    @property
    def condyle_radius(self) -> float:
        """
        Radius of the spherical condyle.
        """

        return self.condyle_diameter / 2.0

    @property
    def posterior_stop_length(self) -> float:
        """
        Physical length of the posterior stop.

        It equals one complete condylar diameter.
        """

        return self.condyle_diameter

    @property
    def maximum_translation(self) -> float:
        """
        Available condylar translation in millimeters.

        One radius is occupied between the corner
        of the L and the initial tangency point.
        """

        return self.length - self.condyle_radius

    @property
    def trajectory_direction(self) -> np.ndarray:
        """
        Unit direction of condylar translation.

        The condyle moves toward +Y and -Z:
        anterior and inferior.
        """

        angle = math.radians(
            self.angle_degrees
        )

        return (
            math.cos(angle)
            * self.coordinate_system.y_axis
            - math.sin(angle)
            * self.coordinate_system.z_axis
        )

    @property
    def surface_normal(self) -> np.ndarray:
        """
        Unit normal of the main guide surface.
        """

        angle = math.radians(
            self.angle_degrees
        )

        return (
            math.sin(angle)
            * self.coordinate_system.y_axis
            + math.cos(angle)
            * self.coordinate_system.z_axis
        )

    @property
    def guide_contact_point(self) -> np.ndarray:
        """
        Initial tangency point between the condyle
        and the main guide surface.
        """

        return (
            self.condyle_center
            + self.condyle_radius
            * self.surface_normal
        )

    @property
    def posterior_stop_normal(self) -> np.ndarray:
        """
        Normal of the posterior stop.

        It points in the permitted direction of
        condylar translation.
        """

        return self.trajectory_direction

    @property
    def posterior_stop_contact_point(
        self,
    ) -> np.ndarray:
        """
        Tangency point between the condyle and the
        posterior stop in centric position.
        """

        return (
            self.condyle_center
            - self.condyle_radius
            * self.trajectory_direction
        )

    @property
    def l_corner(self) -> np.ndarray:
        """
        Common corner of the two surfaces.

        The corner lies one radius behind the main
        contact and one radius above the posterior
        stop contact.
        """

        return (
            self.condyle_center
            + self.condyle_radius
            * self.surface_normal
            - self.condyle_radius
            * self.trajectory_direction
        )

    @property
    def main_surface_vertices(self) -> np.ndarray:
        """
        Four vertices of the main 20 x 20 mm surface.
        """

        half_width = (
            self.width
            / 2.0
            * self.coordinate_system.x_axis
        )

        start = self.l_corner

        end = (
            start
            + self.length
            * self.trajectory_direction
        )

        return np.array([
            start - half_width,
            start + half_width,
            end + half_width,
            end - half_width,
        ])

    @property
    def posterior_stop_vertices(self) -> np.ndarray:
        """
        Four vertices of the posterior stop.

        Its length equals the condylar diameter and
        its width equals the main surface width.
        """

        half_width = (
            self.width
            / 2.0
            * self.coordinate_system.x_axis
        )

        start = self.l_corner

        end = (
            start
            - self.posterior_stop_length
            * self.surface_normal
        )

        return np.array([
            start - half_width,
            start + half_width,
            end + half_width,
            end - half_width,
        ])

    def center_at(
        self,
        distance: float,
    ) -> np.ndarray:
        """
        Returns the condylar center after translation
        along the guide.
        """

        distance = float(distance)

        if (
            not math.isfinite(distance)
            or not 0.0
            <= distance
            <= self.maximum_translation
        ):
            raise ValueError(
                "Translation distance must be between "
                "zero and the available guide length."
            )

        return (
            self.condyle_center
            + distance
            * self.trajectory_direction
        )

    def guide_contact_at(
        self,
        distance: float,
    ) -> np.ndarray:
        """
        Returns the contact point on the main guide
        surface at a translation distance.
        """

        return (
            self.center_at(distance)
            + self.condyle_radius
            * self.surface_normal
        )