"""
OGDD - Guided Protrusion

Rigid bilateral protrusive movement constrained by
two parallel condylar guides.
"""

from dataclasses import dataclass
import math

import numpy as np

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.mesh import Mesh
from ogdd.articulator.condylar_guide import (
    CondylarGuide,
)
from ogdd.geometry.transform import Transform


@dataclass(frozen=True)
class MandibularProtrusivePosition:
    """
    Complete mandibular position during protrusion.
    """

    mesh: Mesh

    balkwill: BalkwillTriangle

    bonwill: BonwillTriangle

    hinge_axis: HingeAxis

    distance_mm: float

@dataclass(frozen=True)
class GuidedProtrusion:
    """
    Bilateral protrusion constrained by condylar guides.

    Both condyles advance and descend the same distance
    along parallel trajectories. The complete mandible
    therefore moves through one rigid translation that
    preserves the original intercondylar distance.
    """

    hinge_axis: HingeAxis
    right_guide: CondylarGuide
    left_guide: CondylarGuide

    def __post_init__(self) -> None:
        """
        Validate the hinge axis and condylar guides.
        """

        if not np.allclose(
            self.right_guide.condyle_center,
            self.hinge_axis.right_condyle.point,
        ):
            raise ValueError(
                "The right guide must belong to the "
                "right condyle."
            )

        if not np.allclose(
            self.left_guide.condyle_center,
            self.hinge_axis.left_condyle.point,
        ):
            raise ValueError(
                "The left guide must belong to the "
                "left condyle."
            )

        right_axis_dot = np.dot(
            self.right_guide.trajectory_direction,
            self.hinge_axis.direction,
        )

        left_axis_dot = np.dot(
            self.left_guide.trajectory_direction,
            self.hinge_axis.direction,
        )

        if not np.isclose(
            right_axis_dot,
            0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "The right guide trajectory must be "
                "orthogonal to the hinge axis."
            )

        if not np.isclose(
            left_axis_dot,
            0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "The left guide trajectory must be "
                "orthogonal to the hinge axis."
            )

        if not np.allclose(
            self.right_guide.trajectory_direction,
            self.left_guide.trajectory_direction,
            atol=1e-8,
        ):
            raise ValueError(
                "Symmetric protrusion requires "
                "parallel condylar trajectories."
            )

    @property
    def maximum_translation(self) -> float:
        """
        Maximum common guide travel in millimeters.

        The shorter available trajectory determines
        the limit for the bilateral movement.
        """

        return min(
            self.right_guide.maximum_translation,
            self.left_guide.maximum_translation,
        )

    @property
    def trajectory_direction(self) -> np.ndarray:
        """
        Shared unit direction of protrusive movement.
        """

        return (
            self.right_guide.trajectory_direction.copy()
        )

    def validated_distance(
        self,
        distance: float,
    ) -> float:
        """
        Validate a requested protrusive distance.
        """

        distance = float(distance)

        if (
            not math.isfinite(distance)
            or not 0.0
            <= distance
            <= self.maximum_translation
        ):
            raise ValueError(
                "Protrusion distance must be between "
                "zero and the common guide limit."
            )

        return distance

    def translation_vector_at(
        self,
        distance: float,
    ) -> np.ndarray:
        """
        Return the rigid protrusive translation vector.
        """

        distance = self.validated_distance(
            distance
        )

        return (
            distance
            * self.trajectory_direction
        )

    def right_target_at(
        self,
        distance: float,
    ) -> np.ndarray:
        """
        Return the right-condyle guided position.
        """

        distance = self.validated_distance(
            distance
        )

        return self.right_guide.center_at(
            distance
        )

    def left_target_at(
        self,
        distance: float,
    ) -> np.ndarray:
        """
        Return the left-condyle guided position.
        """

        distance = self.validated_distance(
            distance
        )

        return self.left_guide.center_at(
            distance
        )

    def transform_at(
        self,
        distance: float,
    ) -> Transform:
        """
        Build the rigid protrusive transformation.
        """
        translation_vector = (
            self.translation_vector_at(
                distance
            )
        )

        if np.allclose(
            translation_vector,
            np.zeros(3),
            atol=1e-12,
        ):
            return Transform.identity()

        return Transform.translation(
            translation_vector
        )

    def translate_points(

        self,
        points: np.ndarray,
        distance: float,
    ) -> np.ndarray:
        """
        Translate 3D points along both guides.
        """

        return self.transform_at(
            distance=distance,
        ).apply(points)

    def translate_landmark(
        self,
        landmark: Landmark,
        distance: float,
    ) -> Landmark:
        """
        Return a translated anatomical landmark.
        """

        translated_point = self.translate_points(
            points=np.array([
                landmark.point,
            ]),
            distance=distance,
        )[0]

        return Landmark(
            name=landmark.name,
            point=translated_point,
            reference_used=landmark.reference_used,
            confidence=landmark.confidence,
            created_by=landmark.created_by,
        )

    def translate_balkwill(
        self,
        balkwill: BalkwillTriangle,
        distance: float,
    ) -> BalkwillTriangle:
        """
        Translate the complete Balkwill triangle.
        """

        return BalkwillTriangle(
            left_posterior=self.translate_landmark(
                landmark=balkwill.left_posterior,
                distance=distance,
            ),
            right_posterior=self.translate_landmark(
                landmark=balkwill.right_posterior,
                distance=distance,
            ),
            dental_midline=self.translate_landmark(
                landmark=balkwill.dental_midline,
                distance=distance,
            ),
        )

    def translate_bonwill(
        self,
        bonwill: BonwillTriangle,
        distance: float,
    ) -> BonwillTriangle:
        """
        Translate the complete Bonwill triangle.
        """

        return BonwillTriangle(
            left_condyle=self.translate_landmark(
                landmark=bonwill.left_condyle,
                distance=distance,
            ),
            right_condyle=self.translate_landmark(
                landmark=bonwill.right_condyle,
                distance=distance,
            ),
            dental_midline=self.translate_landmark(
                landmark=bonwill.dental_midline,
                distance=distance,
            ),
        )

    def translate_mesh(
        self,
        mesh: Mesh,
        distance: float,
    ) -> Mesh:
        """
        Return a rigidly translated mesh copy.
        """

        transform = self.transform_at(
            distance=distance
        )

        translated_vertices = transform.apply(
            mesh.vertices
        )

        translated_normals = None

        if mesh.normals is not None:
            translated_normals = (
                transform.apply_vectors(
                    mesh.normals
                )
            )

        return Mesh(
            vertices=translated_vertices,
            faces=mesh.faces.copy(),
            normals=translated_normals,
            attributes={
                name: values.copy()
                for name, values
                in mesh.attributes.items()
            },
            metadata=mesh.metadata.copy(),
        )

    def position_at(
        self,
        assembly: MandibularAssembly,
        distance: float,
    ) -> MandibularProtrusivePosition:
        """
        Calculate a complete protrusive position.

        Every position is calculated from the original
        assembly to avoid accumulated translation errors.
        """

        if not np.allclose(
            assembly.hinge_axis.left_condyle.point,
            self.hinge_axis.left_condyle.point,
        ):
            raise ValueError(
                "The assembly and protrusion must share "
                "the same left condyle."
            )

        if not np.allclose(
            assembly.hinge_axis.right_condyle.point,
            self.hinge_axis.right_condyle.point,
        ):
            raise ValueError(
                "The assembly and protrusion must share "
                "the same right condyle."
            )

        distance = self.validated_distance(
            distance
        )

        translated_mesh = self.translate_mesh(
            mesh=assembly.mesh,
            distance=distance,
        )

        translated_balkwill = self.translate_balkwill(
            balkwill=assembly.balkwill,
            distance=distance,
        )

        translated_bonwill = self.translate_bonwill(
            bonwill=assembly.bonwill,
            distance=distance,
        )

        translated_hinge_axis = HingeAxis(
            left_condyle=(
                translated_bonwill.left_condyle
            ),
            right_condyle=(
                translated_bonwill.right_condyle
            ),
        )

        return MandibularProtrusivePosition(
            mesh=translated_mesh,
            balkwill=translated_balkwill,
            bonwill=translated_bonwill,
            hinge_axis=translated_hinge_axis,
            distance_mm=distance,
        )