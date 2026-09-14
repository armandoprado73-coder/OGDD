"""Tests for restoring virtual articulator controls."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from ogdd.app.mounting_panel import MountingPanel
from ogdd.articulator.configuration import ArticulatorConfiguration


@pytest.fixture(scope="module")
def qt_application():
    return QApplication.instance() or QApplication([])


def test_saved_configuration_returns_to_controls(qt_application) -> None:
    panel = MountingPanel()
    expected = ArticulatorConfiguration(
        intercondylar_width=140.0,
        balkwill_angle_degrees=27.0,
        right_condylar_guidance_degrees=42.0,
        left_condylar_guidance_degrees=48.0,
    )

    panel.set_configuration(expected)

    assert panel.configuration() == expected


def test_unknown_width_is_not_silently_changed(qt_application) -> None:
    panel = MountingPanel()
    configuration = ArticulatorConfiguration(intercondylar_width=120.0)

    with pytest.raises(ValueError, match="preset"):
        panel.set_configuration(configuration)
