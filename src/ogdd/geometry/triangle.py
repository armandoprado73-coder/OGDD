"""
OGDD - Triangle Geometry

Representa un triángulo en el espacio 3D.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Triangle:
    """
    Triángulo definido por tres puntos en 3D.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


    @property
    def side_ab(self) -> float:
        """
        Distancia entre los puntos A y B.
        """

        return np.linalg.norm(self.b - self.a)


    @property
    def side_bc(self) -> float:
        """
        Distancia entre los puntos B y C.
        """

        return np.linalg.norm(self.c - self.b)


    @property
    def side_ca(self) -> float:
        """
        Distancia entre los puntos C y A.
        """

        return np.linalg.norm(self.a - self.c)


    @property
    def centroid(self) -> np.ndarray:
        """
        Centroide del triángulo.
        """

        return (self.a + self.b + self.c) / 3