"""
OGDD - Dental Model

Representa un modelo digital de una arcada dental.
"""

from dataclasses import dataclass, field

from ogdd.mesh import Mesh

from .landmark import Landmark
from .balkwill import BalkwillTriangle
from .bonwill import BonwillTriangle

@dataclass
class DentalModel:
    """
    Modelo anatómico de una arcada dental.
    """

    mesh: Mesh

    landmarks: dict[str, Landmark] = field(default_factory=dict)


    def add_landmark(self, landmark: Landmark) -> None:
        """
        Agrega una referencia anatómica al modelo.
        """

        if landmark.name in self.landmarks:
            raise ValueError(
                f"Landmark '{landmark.name}' already exists."
            )

        self.landmarks[landmark.name] = landmark


    def get_landmark(self, name: str) -> Landmark | None:
        """
        Devuelve un landmark por su nombre.
        """

        return self.landmarks.get(name)


    def remove_landmark(self, name: str) -> None:
        """
        Elimina un landmark del modelo.
        """

        self.landmarks.pop(name, None)

    @property
    def landmark_count(self) -> int:
        """
        Número de referencias anatómicas.
        """

        return len(self.landmarks)

    @property
    def is_balkwill_ready(self) -> bool:
        """
        Indica si están definidos los tres puntos necesarios
        para construir el triángulo de Balkwill.
        """

        required = {
            "RIGHT_SECOND_MOLAR",
            "LEFT_SECOND_MOLAR",
            "DENTAL_MIDLINE",
        }

        return required.issubset(self.landmarks.keys())


    @property
    def balkwill_triangle(self) -> BalkwillTriangle:
        """
        Construye el triángulo de Balkwill
        a partir de los landmarks anatómicos.
        """

        if not self.is_balkwill_ready:
            raise ValueError(
                "DentalModel is not ready for Balkwill triangle."
            )

        return BalkwillTriangle(
            left_posterior=self.landmarks["LEFT_SECOND_MOLAR"],
            right_posterior=self.landmarks["RIGHT_SECOND_MOLAR"],
            dental_midline=self.landmarks["DENTAL_MIDLINE"],
        )

    @property
    def is_bonwill_ready(self) -> bool:
        """
        Indica si están definidos los tres puntos necesarios
        para construir el triángulo de Bonwill.
        """

        required = {
            "RIGHT_CONDYLE",
            "LEFT_CONDYLE",
            "DENTAL_MIDLINE",
        }

        return required.issubset(self.landmarks.keys())


    @property
    def bonwill_triangle(self) -> BonwillTriangle:
        """
        Construye el triángulo de Bonwill
        a partir de los landmarks anatómicos.
        """

        if not self.is_bonwill_ready:
            raise ValueError(
                "DentalModel is not ready for Bonwill triangle."
            )

        return BonwillTriangle(
            left_condyle=self.landmarks["LEFT_CONDYLE"],
            right_condyle=self.landmarks["RIGHT_CONDYLE"],
            dental_midline=self.landmarks["DENTAL_MIDLINE"],
        )