"""
OGDD PCA Tests

Tests for principal component analysis
on 3D point clouds.
"""

import numpy as np
import pytest

from ogdd.algorithms.pca import (
    PrincipalComponentAnalysis
)



def test_pca_centroid():
    """
    Test centroid calculation.
    """

    points = np.array(
        [
            [0,0,0],
            [2,0,0],
            [4,0,0]
        ],
        dtype=float
    )


    result = PrincipalComponentAnalysis.compute(
        points
    )


    assert np.allclose(
        result.centroid,
        [2,0,0]
    )



def test_pca_detects_main_axis():
    """
    Test principal direction.

    Points aligned with X axis.
    """

    points = np.array(
        [
            [-5,0,0],
            [-2,0,0],
            [0,0,0],
            [3,0,0],
            [7,0,0]
        ],
        dtype=float
    )


    result = PrincipalComponentAnalysis.compute(
        points
    )


    axis = result.principal_axis


    # Direction can be positive or negative.
    assert np.isclose(
        abs(axis[0]),
        1.0
    )



def test_pca_axes_are_orthogonal():
    """
    Test PCA coordinate system.
    """

    points = np.array(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [2,5,1],
            [9,3,4]
        ],
        dtype=float
    )


    result = PrincipalComponentAnalysis.compute(
        points
    )


    dot_product = np.dot(
        result.principal_axis,
        result.secondary_axis
    )


    assert np.isclose(
        dot_product,
        0,
        atol=1e-7
    )



def test_eigenvalues_are_ordered():
    """
    Test descending variance order.
    """

    points = np.random.rand(
        100,
        3
    )


    result = PrincipalComponentAnalysis.compute(
        points
    )


    assert (
        result.eigenvalues[0]
        >=
        result.eigenvalues[1]
    )


    assert (
        result.eigenvalues[1]
        >=
        result.eigenvalues[2]
    )



def test_invalid_point_dimension():

    with pytest.raises(ValueError):

        PrincipalComponentAnalysis.compute(
            [
                [1,2],
                [3,4]
            ]
        )



def test_not_enough_points():

    with pytest.raises(ValueError):

        PrincipalComponentAnalysis.compute(
            [
                [0,0,0],
                [1,1,1]
            ]
        )
