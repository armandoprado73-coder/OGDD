"""
OGDD - Coordinate System

Sistema de coordenadas 3D y anatómico para OGDD.

Convención anatómica OGDD:

    +X = derecha del paciente
    +Y = anterior
    +Z = superior

    Origen anatómico = línea media dental
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CoordinateSystem:
    """
    Sistema de coordenadas ortonormal en 3D.
    """

    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray

    def __post_init__(self):
        """
        Convierte los datos a arrays NumPy y normaliza los ejes.
        """

        self.origin = np.asarray(
            self.origin,
            dtype=float,
        )

        self.x_axis = np.asarray(
            self.x_axis,
            dtype=float,
        )

        self.y_axis = np.asarray(
            self.y_axis,
            dtype=float,
        )

        self.z_axis = np.asarray(
            self.z_axis,
            dtype=float,
        )

        # ----------------------------------------------------
        # Validación de dimensiones
        # ----------------------------------------------------

        if self.origin.shape != (3,):
            raise ValueError(
                "Origin must be a 3D point."
            )

        if self.x_axis.shape != (3,):
            raise ValueError(
                "X axis must be a 3D vector."
            )

        if self.y_axis.shape != (3,):
            raise ValueError(
                "Y axis must be a 3D vector."
            )

        if self.z_axis.shape != (3,):
            raise ValueError(
                "Z axis must be a 3D vector."
            )

        # ----------------------------------------------------
        # Normalización
        # ----------------------------------------------------

        x_norm = np.linalg.norm(self.x_axis)
        y_norm = np.linalg.norm(self.y_axis)
        z_norm = np.linalg.norm(self.z_axis)

        if np.isclose(x_norm, 0.0):
            raise ValueError(
                "X axis cannot have zero length."
            )

        if np.isclose(y_norm, 0.0):
            raise ValueError(
                "Y axis cannot have zero length."
            )

        if np.isclose(z_norm, 0.0):
            raise ValueError(
                "Z axis cannot have zero length."
            )

        self.x_axis = self.x_axis / x_norm
        self.y_axis = self.y_axis / y_norm
        self.z_axis = self.z_axis / z_norm

    # ========================================================
    # CONSTRUCCIÓN GENERAL
    # ========================================================

    @classmethod
    def from_three_points(
        cls,
        origin,
        point_x,
        point_y,
    ) -> "CoordinateSystem":
        """
        Construye un sistema de coordenadas a partir de tres puntos.

        origin
            Punto de origen.

        point_x
            Define la dirección de +X.

        point_y
            Define la dirección aproximada de +Y.

        El eje Y se ortogonaliza respecto a X mediante
        Gram-Schmidt.

        El eje Z se obtiene mediante:

            Z = X × Y
        """

        origin = np.asarray(
            origin,
            dtype=float,
        )

        point_x = np.asarray(
            point_x,
            dtype=float,
        )

        point_y = np.asarray(
            point_y,
            dtype=float,
        )

        if origin.shape != (3,):
            raise ValueError(
                "Origin must be a 3D point."
            )

        if point_x.shape != (3,):
            raise ValueError(
                "point_x must be a 3D point."
            )

        if point_y.shape != (3,):
            raise ValueError(
                "point_y must be a 3D point."
            )

        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        x_vector = point_x - origin

        x_norm = np.linalg.norm(x_vector)

        if np.isclose(x_norm, 0.0):
            raise ValueError(
                "point_x cannot be equal to origin."
            )

        x_axis = x_vector / x_norm

        # ----------------------------------------------------
        # Y anatómico aproximado
        # ----------------------------------------------------

        y_vector = point_y - origin

        # ----------------------------------------------------
        # Gram-Schmidt
        # ----------------------------------------------------

        y_orthogonal = (
            y_vector
            - np.dot(y_vector, x_axis) * x_axis
        )

        y_norm = np.linalg.norm(
            y_orthogonal
        )

        if np.isclose(y_norm, 0.0):
            raise ValueError(
                "The three points are collinear."
            )

        y_axis = (
            y_orthogonal
            / y_norm
        )

        # ----------------------------------------------------
        # Z
        # ----------------------------------------------------

        z_axis = np.cross(
            x_axis,
            y_axis,
        )

        z_norm = np.linalg.norm(z_axis)

        if np.isclose(z_norm, 0.0):
            raise ValueError(
                "Cannot construct Z axis."
            )

        z_axis = z_axis / z_norm

        return cls(
            origin=origin,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
        )

    # ========================================================
    # CONSTRUCCIÓN ANATÓMICA DENTAL
    # ========================================================

    @classmethod
    def from_dental_landmarks(
        cls,
        right_molar,
        left_molar,
        dental_midline,
    ) -> "CoordinateSystem":
        """
        Construye el sistema anatómico de OGDD.

        Convención:

            Origen = línea media dental
            +X = derecha del paciente
            +Y = anterior
            +Z = superior

        Los molares determinan la dirección transversal.
        La línea media determina la dirección anterior.
        """

        right_molar = np.asarray(
            right_molar,
            dtype=float,
        )

        left_molar = np.asarray(
            left_molar,
            dtype=float,
        )

        dental_midline = np.asarray(
            dental_midline,
            dtype=float,
        )

        # ----------------------------------------------------
        # Centro intermolar
        # ----------------------------------------------------

        intermolar_center = (
            right_molar + left_molar
        ) / 2.0

        # ----------------------------------------------------
        # +X = centro intermolar → molar derecho
        # ----------------------------------------------------

        x_vector = (
            right_molar
            - intermolar_center
        )

        x_norm = np.linalg.norm(x_vector)

        if np.isclose(x_norm, 0.0):
            raise ValueError(
                "Molar points cannot define an X axis."
            )

        x_axis = x_vector / x_norm

        # ----------------------------------------------------
        # Dirección anatómica hacia anterior
        #
        # centro intermolar → línea media
        # ----------------------------------------------------

        y_vector = (
            dental_midline
            - intermolar_center
        )

        # ----------------------------------------------------
        # Ortogonalización de Y respecto a X
        # ----------------------------------------------------

        y_orthogonal = (
            y_vector
            - np.dot(y_vector, x_axis) * x_axis
        )

        y_norm = np.linalg.norm(
            y_orthogonal
        )

        if np.isclose(y_norm, 0.0):
            raise ValueError(
                "Dental landmarks cannot define "
                "anterior direction."
            )

        y_axis = (
            y_orthogonal
            / y_norm
        )

        # ----------------------------------------------------
        # +Z = +X × +Y
        # ----------------------------------------------------

        z_axis = np.cross(
            x_axis,
            y_axis,
        )

        z_norm = np.linalg.norm(z_axis)

        if np.isclose(z_norm, 0.0):
            raise ValueError(
                "Dental landmarks cannot define Z axis."
            )

        z_axis = z_axis / z_norm

        return cls(
            origin=dental_midline,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
        )

    # ========================================================
    # SISTEMA IDENTIDAD
    # ========================================================

    @classmethod
    def identity(cls) -> "CoordinateSystem":
        """
        Devuelve el sistema de coordenadas mundial estándar.
        """

        return cls(
            origin=np.array(
                [0.0, 0.0, 0.0]
            ),
            x_axis=np.array(
                [1.0, 0.0, 0.0]
            ),
            y_axis=np.array(
                [0.0, 1.0, 0.0]
            ),
            z_axis=np.array(
                [0.0, 0.0, 1.0]
            ),
        )

    # ========================================================
    # WORLD → LOCAL
    # ========================================================

    def to_local(
        self,
        points,
    ) -> np.ndarray:
        """
        Convierte puntos del sistema mundial
        al sistema local.
        """

        points = np.asarray(
            points,
            dtype=float,
        )

        relative = points - self.origin

        rotation = np.column_stack(
            [
                self.x_axis,
                self.y_axis,
                self.z_axis,
            ]
        )

        return relative @ rotation

    # ========================================================
    # LOCAL → WORLD
    # ========================================================

    def to_world(
        self,
        points,
    ) -> np.ndarray:
        """
        Convierte puntos del sistema local
        al sistema mundial.
        """

        points = np.asarray(
            points,
            dtype=float,
        )

        rotation = np.column_stack(
            [
                self.x_axis,
                self.y_axis,
                self.z_axis,
            ]
        )

        return points @ rotation.T + self.origin