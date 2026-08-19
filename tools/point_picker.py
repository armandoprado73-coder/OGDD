"""
OGDD - 3D Anatomical Point Picker

Herramienta auxiliar para seleccionar landmarks anatómicos
sobre una malla 3D y construir el triángulo de Balkwill.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from ogdd.anatomy.dental_model import DentalModel
from ogdd.anatomy.landmark import Landmark
from ogdd.io.stl import STLReader


# ------------------------------------------------------------
# Archivo STL de prueba
# ------------------------------------------------------------

STL_PATH = Path(
    "data/stl/Mandibular Anatomy_Mordida normal.stl"
)


# ------------------------------------------------------------
# Cargar modelo OGDD
# ------------------------------------------------------------

mesh = STLReader.read(STL_PATH)

model = DentalModel(mesh)


# ------------------------------------------------------------
# Cargar modelo para visualización
# ------------------------------------------------------------

display_mesh = pv.read(STL_PATH)

print(f"Modelo cargado: {STL_PATH}")
print(f"Puntos de la malla: {display_mesh.n_points}")
print(f"Caras de la malla: {display_mesh.n_cells}")


# ------------------------------------------------------------
# Orden de selección anatómica
# ------------------------------------------------------------

LANDMARK_SEQUENCE = [
    (
        "RIGHT_SECOND_MOLAR",
        "RIGHT MOLAR",
        "Right second molar cusp",
    ),
    (
        "LEFT_SECOND_MOLAR",
        "LEFT MOLAR",
        "Left second molar cusp",
    ),
    (
        "DENTAL_MIDLINE",
        "DENTAL MIDLINE",
        "Dental midline",
    ),
]

selection_index = 0


# ------------------------------------------------------------
# Crear visor
# ------------------------------------------------------------

plotter = pv.Plotter()

plotter.add_mesh(
    display_mesh,
    show_edges=False,
)

plotter.add_text(
    "Select RIGHT SECOND MOLAR",
    position="upper_left",
    font_size=12,
    name="instruction",
)


# ------------------------------------------------------------
# Mostrar Balkwill
# ------------------------------------------------------------

def display_balkwill():
    """
    Construye y muestra el triángulo de Balkwill
    después de seleccionar los tres landmarks.
    """

    balkwill = model.balkwill_triangle

    right = balkwill.right_posterior.point
    left = balkwill.left_posterior.point
    midline = balkwill.dental_midline.point

    # --------------------------------------------------------
    # Dibujar triángulo
    # --------------------------------------------------------

    plotter.add_mesh(
        pv.Line(right, left),
        line_width=5,
        color="yellow",
    )

    plotter.add_mesh(
        pv.Line(left, midline),
        line_width=5,
        color="yellow",
    )

    plotter.add_mesh(
        pv.Line(midline, right),
        line_width=5,
        color="yellow",
    )

    # --------------------------------------------------------
    # Sistema anatómico
    # --------------------------------------------------------

    coordinate_system = model.coordinate_system

    origin = np.array(
        [coordinate_system.origin]
    )

    axis_length = 15.0

    plotter.add_arrows(
        origin,
        np.array([coordinate_system.x_axis]),
        mag=axis_length,
        color="red",
    )

    plotter.add_arrows(
        origin,
        np.array([coordinate_system.y_axis]),
        mag=axis_length,
        color="green",
    )

    plotter.add_arrows(
        origin,
        np.array([coordinate_system.z_axis]),
        mag=axis_length,
        color="blue",
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print("BALKWILL TRIANGLE")
    print("=" * 50)

    print(
        f"Right side          : "
        f"{balkwill.right_side:.6f}"
    )

    print(
        f"Left side           : "
        f"{balkwill.left_side:.6f}"
    )

    print(
        f"Intermolar width    : "
        f"{balkwill.intermolar_width:.6f}"
    )

    print(
        f"Symmetry difference : "
        f"{balkwill.symmetry_difference:.6f}"
    )

    print()
    print("ANATOMICAL COORDINATE SYSTEM")
    print("-" * 30)

    print(f"Origin : {coordinate_system.origin}")
    print(f"+X     : {coordinate_system.x_axis}")
    print(f"+Y     : {coordinate_system.y_axis}")
    print(f"+Z     : {coordinate_system.z_axis}")

    plotter.add_text(
        "Balkwill ready",
        position="upper_left",
        font_size=12,
        name="instruction",
    )


# ------------------------------------------------------------
# Selección de landmarks
# ------------------------------------------------------------

def pick_point(point):
    """
    Registra los landmarks anatómicos
    en el orden establecido.
    """

    global selection_index

    if point is None:
        return

    if selection_index >= len(LANDMARK_SEQUENCE):
        print()
        print("Anatomical landmarks already selected.")
        return

    name, label, reference = (
        LANDMARK_SEQUENCE[selection_index]
    )

    point = np.asarray(
        point,
        dtype=float,
    )

    landmark = Landmark(
        name=name,
        point=point,
        reference_used=reference,
    )

    model.add_landmark(landmark)

    print()
    print(f"{label} selected:")
    print(f"X = {point[0]:.6f}")
    print(f"Y = {point[1]:.6f}")
    print(f"Z = {point[2]:.6f}")

    plotter.add_point_labels(
        np.array([point]),
        [label],
        point_size=12,
        font_size=12,
        render_points_as_spheres=True,
    )

    selection_index += 1

    # --------------------------------------------------------
    # Indicar siguiente landmark
    # --------------------------------------------------------

    if selection_index < len(LANDMARK_SEQUENCE):

        next_label = (
            LANDMARK_SEQUENCE[selection_index][1]
        )

        plotter.add_text(
            f"Select {next_label}",
            position="upper_left",
            font_size=12,
            name="instruction",
        )

    # --------------------------------------------------------
    # Los tres landmarks ya existen
    # --------------------------------------------------------

    else:

        print()
        print("Three anatomical landmarks selected.")
        print(f"Balkwill ready: {model.is_balkwill_ready}")
        print(
            f"Coordinate system ready: "
            f"{model.is_coordinate_system_ready}"
        )

        display_balkwill()


# ------------------------------------------------------------
# Activar selección
# ------------------------------------------------------------

plotter.enable_surface_point_picking(
    callback=pick_point,
    show_point=True,
    show_message=False,
)


# ------------------------------------------------------------
# Mostrar
# ------------------------------------------------------------

plotter.show()