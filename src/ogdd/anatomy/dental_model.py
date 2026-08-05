"""
OGDD - Dental Model

Representa un modelo digital de una arcada dental.
"""

from dataclasses import dataclass, field

from ogdd.mesh import Mesh

from .landmark import Landmark


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