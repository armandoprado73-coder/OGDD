"""
OGDD Vector Mathematics

Basic vector operations used throughout
the computational geometry engine.

All functions operate on NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def magnitude(vector: np.ndarray) -> float:
    """
    Calculate vector length.

    Parameters
    ----------
    vector:
        Vector with shape (3,)

    Returns
    -------
    float
        Euclidean magnitude.
    """

    vector = np.asarray(vector, dtype=float)

    return float(np.linalg.norm(vector))



def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector.

    Parameters
    ----------
    vector:
        Input vector.

    Returns
    -------
    numpy.ndarray
        Unit vector.

    Raises
    ------
    ValueError
        If vector magnitude is zero.
    """

    vector = np.asarray(
        vector,
        dtype=float
    )

    length = np.linalg.norm(vector)

    if length == 0:
        raise ValueError(
            "Cannot normalize zero vector."
        )

    return vector / length



def dot(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Calculate dot product.

    Returns:

        a · b
    """

    return float(
        np.dot(vector_a, vector_b)
    )



def cross(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> np.ndarray:
    """
    Calculate cross product.

    Returns:

        a × b
    """

    return np.cross(
        vector_a,
        vector_b
    )



def angle_between(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Calculate angle between two vectors.

    Result is expressed in radians.

    Formula:

        cos(theta)=a·b/(|a||b|)
    """

    a = normalize(vector_a)

    b = normalize(vector_b)

    cosine = np.clip(
        dot(a, b),
        -1.0,
        1.0
    )

    return float(
        np.arccos(cosine)
    )
