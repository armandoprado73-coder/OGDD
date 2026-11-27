"""Clinical direction, magnitude, and recorded-case MCD regressions."""

import numpy as np
import pytest

from ogdd.geometry.mcd_axes import anatomical_to_mcd


@pytest.mark.parametrize("anatomical, expected", [
    ([1, 0, 0], [0, 1, 0]),       # patient right
    ([-1, 0, 0], [0, -1, 0]),     # patient left
    ([0, 1, 0], [1, 0, 0]),       # anterior
    ([0, -1, 0], [-1, 0, 0]),     # posterior
    ([0, 0, 1], [0, 0, -1]),      # superior
    ([0, 0, -1], [0, 0, 1]),      # inferior
    ([0.9136, 1.3235, 0.4433], [1.3235, 0.9136, -0.4433]),
    ([0.9406, -0.9251, -0.4932], [-0.9251, 0.9406, 0.4932]),
    ([0.2056, -0.1554, -1.3767], [-0.1554, 0.2056, 1.3767]),
    ([0.2063, -0.3132, -1.7357], [-0.3132, 0.2063, 1.7357]),
])
def test_clinical_directions_and_case_components(anatomical, expected):
    source = np.array(anatomical, dtype=float)
    original = source.copy()
    converted = anatomical_to_mcd(source)
    np.testing.assert_array_equal(converted, expected)
    np.testing.assert_array_equal(source, original)
    assert np.linalg.norm(converted) == pytest.approx(np.linalg.norm(source))
    assert not np.shares_memory(source, converted)


@pytest.mark.parametrize("invalid", [[1, 2], [[1, 2, 3]],
                                     [1, np.nan, 3], [1, 2, np.inf]])
def test_invalid_displacement_is_rejected(invalid):
    with pytest.raises(ValueError):
        anatomical_to_mcd(invalid)


def test_zero_components_do_not_display_negative_zero():
    result = anatomical_to_mcd([0, 0, 0])
    assert not np.any(np.signbit(result))
