"""
OGDD Vector Mathematics Tests

Tests for the fundamental vector operations
used by the geometry engine.
"""

import numpy as np
import pytest

from ogdd.geometry.vectors import (
    magnitude,
    normalize,
    dot,
    cross,
    angle_between,
)



def test_vector_magnitude():
    """
    Test Euclidean vector length.
    """

    vector = np.array(
        [3.0, 4.0, 0.0]
    )

    result = magnitude(vector)

    assert result == 5.0



def test_vector_normalization():
    """
    Test unit vector generation.
    """

    vector = np.array(
        [2.0, 0.0, 0.0]
    )

    result = normalize(vector)

    expected = np.array(
        [1.0, 0.0, 0.0]
    )

    assert np.allclose(
        result,
        expected
    )



def test_zero_vector_normalization():
    """
    Zero vector cannot be normalized.
    """

    vector = np.array(
        [0.0, 0.0, 0.0]
    )

    with pytest.raises(ValueError):

        normalize(vector)



def test_dot_product():
    """
    Test scalar product.
    """

    a = np.array(
        [1.0, 0.0, 0.0]
    )

    b = np.array(
        [0.0, 1.0, 0.0]
    )


    result = dot(a, b)

    assert result == 0.0



def test_cross_product():
    """
    Test vector product.
    """

    a = np.array(
        [1.0, 0.0, 0.0]
    )

    b = np.array(
        [0.0, 1.0, 0.0]
    )


    result = cross(a, b)

    expected = np.array(
        [0.0, 0.0, 1.0]
    )


    assert np.allclose(
        result,
        expected
    )



def test_angle_between_vectors():
    """
    Test angle calculation.
    """

    a = np.array(
        [1.0, 0.0, 0.0]
    )

    b = np.array(
        [0.0, 1.0, 0.0]
    )


    result = angle_between(
        a,
        b
    )


    assert np.isclose(
        result,
        np.pi / 2
    )
