"""
OGDD - Landmark

Represents an anatomical reference identified
on a dental model.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Landmark:
    """
    Anatomical reference identified on a dental model.

    Parameters
    ----------
    name: str
        Unique landmark identifier.

    point: Np.ndarray
        Landmark position in 3D space.

    reference_used : str
        Anatomical reference actually used by the
        operator or algorithm.

    confidence : float, default=1.0
        Confidence score between 0.0 and 1.0.

    created_by : str, default="operator"
        Creator of the landmark.
    """

    name: str

    point: np.ndarray

    reference_used: str

    confidence: float = 1.0

    created_by: str = "operator"