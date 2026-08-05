"""
Tests for Triangle geometry.
"""

import numpy as np

from ogdd.geometry.triangle import Triangle



def test_triangle_side_lengths():
    """
    Test side length calculations.
    """

    triangle = Triangle(
        a=np.array([0.0, 0.0, 0.0]),
        b=np.array([3.0, 0.0, 0.0]),
        c=np.array([0.0, 4.0, 0.0]),
    )


    assert triangle.side_ab == 3.0
    assert triangle.side_bc == 5.0
    assert triangle.side_ca == 4.0



def test_triangle_centroid():
    """
    Test centroid calculation.
    """

    triangle = Triangle(
        a=np.array([0.0, 0.0, 0.0]),
        b=np.array([3.0, 0.0, 0.0]),
        c=np.array([0.0, 3.0, 0.0]),
    )


    centroid = triangle.centroid


    assert np.allclose(
        centroid,
        np.array([1.0, 1.0, 0.0])
    )