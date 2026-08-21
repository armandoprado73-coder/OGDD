"""
Tests for virtual articulator configuration.
"""

import pytest

from ogdd.articulator.configuration import (
    ArticulatorConfiguration,
    IntercondylarPreset,
)


def test_default_intercondylar_width_is_medium():

    configuration = ArticulatorConfiguration()

    assert configuration.intercondylar_width == 110.0


def test_small_intercondylar_preset():

    configuration = ArticulatorConfiguration.from_preset(
        IntercondylarPreset.S
    )

    assert configuration.intercondylar_width == 95.0


def test_medium_intercondylar_preset():

    configuration = ArticulatorConfiguration.from_preset(
        IntercondylarPreset.M
    )

    assert configuration.intercondylar_width == 110.0


def test_large_intercondylar_preset():

    configuration = ArticulatorConfiguration.from_preset(
        IntercondylarPreset.L
    )

    assert configuration.intercondylar_width == 140.0


def test_intercondylar_width_must_be_positive():

    with pytest.raises(ValueError):
        ArticulatorConfiguration(
            intercondylar_width=0.0
        )

def test_default_balkwill_angle_is_25_degrees():

    configuration = ArticulatorConfiguration()

    assert configuration.balkwill_angle_degrees == 25.0


def test_bonwill_side_uses_articulator_size():

    configuration = ArticulatorConfiguration.from_preset(
        IntercondylarPreset.M
    )

    assert configuration.bonwill_side_length == 110.0


def test_balkwill_angle_must_be_valid():

    with pytest.raises(ValueError):
        ArticulatorConfiguration(
            balkwill_angle_degrees=0.0
        )