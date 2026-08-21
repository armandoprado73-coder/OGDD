"""
OGDD - Bonwill Builder

Builds a virtual Bonwill triangle from an anatomical
coordinate system and an articulator configuration.
"""

from math import cos, radians, sin, sqrt

import numpy as np

from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.landmark import Landmark
from ogdd.geometry.coordinate_system import CoordinateSystem

from .configuration import ArticulatorConfiguration


class BonwillBuilder:
    """
    Builds a virtual Bonwill triangle in relation
    to the patient's anatomical coordinate system.
    """

    @staticmethod
    def build(
        coordinate_system: CoordinateSystem,
        dental_midline: Landmark,
        configuration: ArticulatorConfiguration,
    ) -> BonwillTriangle:
        """
        Builds virtual condylar landmarks and returns
        the corresponding Bonwill triangle.
        """

        side = configuration.bonwill_side_length

        half_width = (
            configuration.intercondylar_width / 2.0
        )

        triangle_height = (
            sqrt(3.0) / 2.0
        ) * side

        angle = radians(
            configuration.balkwill_angle_degrees
        )

        posterior = (
            -triangle_height * cos(angle)
        )

        superior = (
            triangle_height * sin(angle)
        )

        right_local = np.array(
            [
                half_width,
                posterior,
                superior,
            ],
            dtype=float,
        )

        left_local = np.array(
            [
                -half_width,
                posterior,
                superior,
            ],
            dtype=float,
        )

        condyles_world = coordinate_system.to_world(
            np.array(
                [
                    right_local,
                    left_local,
                ]
            )
        )

        right_condyle = Landmark(
            name="RIGHT_CONDYLE",
            point=condyles_world[0],
            reference_used="Virtual articulator",
        )

        left_condyle = Landmark(
            name="LEFT_CONDYLE",
            point=condyles_world[1],
            reference_used="Virtual articulator",
        )

        return BonwillTriangle(
            left_condyle=left_condyle,
            right_condyle=right_condyle,
            dental_midline=dental_midline,
        )