"""Clinical AD2 MCD components of an anatomical RC-to-MIC displacement.

Internal geometry keeps +X patient right, +Y anterior, +Z superior.
Only clinical results use MCD X sagittal (+anterior), Y transverse
(+patient right), and Z vertical (+inferior). This does not relocate the
virtual hinge axis or reproduce the MCD's central transverse table.
"""

from __future__ import annotations

import numpy as np


MCD_AXES_DESCRIPTION = (
    "Ejes MCD (AD2): +X anterior, +Y derecha del paciente, +Z inferior."
)


def anatomical_to_mcd(vector) -> np.ndarray:
    """Convert one anatomical displacement without changing its magnitude.

    Input must already be a vector in the anatomical frame, not a point
    in scanner/world coordinates. Conversion returns a new array and
    must be applied exactly once at the clinical presentation boundary.
    """

    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("Displacement must be a finite three-component vector.")
    result = vector[[1, 0, 2]].copy()
    result[2] *= -1
    result[result == 0] = 0.0
    return result
