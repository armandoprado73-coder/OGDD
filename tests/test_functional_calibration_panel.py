"""Tests for the functional calibration workflow panel."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from ogdd.app.functional_calibration_panel import FunctionalCalibrationPanel


@pytest.fixture(scope="module")
def qt_application():
    """Provide the one QApplication required by all panel tests."""

    return QApplication.instance() or QApplication([])


def test_panel_starts_protected(qt_application) -> None:
    panel = FunctionalCalibrationPanel()

    assert not panel.availability_label.isHidden()
    assert panel.reference_value.text() == "Pendiente"
    assert panel.path_value.text() == "—"
    assert not panel.advance_button.isEnabled()
    assert not panel.retreat_button.isEnabled()


def test_panel_reports_confirmed_rc_mounting(qt_application) -> None:
    panel = FunctionalCalibrationPanel()

    panel.set_mounting_available(True, 17.0)

    assert panel.availability_label.isHidden()
    assert panel.reference_value.text() == "RC confirmada"
    assert panel.path_value.text() == "17.0 mm"
    assert "Apertura" in panel.state_label.text()
    assert "protrusión" in panel.state_label.text()
    assert panel.open_button.isEnabled()
    assert not panel.close_button.isEnabled()
    assert panel.advance_button.isEnabled()
    assert not panel.retreat_button.isEnabled()
    assert not panel.rc_button.isEnabled()


def test_panel_can_report_mounting_without_known_path(qt_application) -> None:
    panel = FunctionalCalibrationPanel()

    panel.set_mounting_available(True)

    assert panel.reference_value.text() == "RC confirmada"
    assert panel.path_value.text() == "—"
    assert "recorrido conocido" in panel.state_label.text()


def test_panel_returns_to_protected_state(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0)

    panel.set_mounting_available(False)

    assert not panel.availability_label.isHidden()
    assert panel.reference_value.text() == "Pendiente"
    assert panel.opening_value.text() == "0.0°"
    assert panel.protrusion_value.text() == "0.0 mm"
    assert panel.lateral_value.text() == "0.0°"
    assert panel.closure_value.text() == "0.0°"
    assert panel.path_value.text() == "—"
    assert not panel.open_button.isEnabled()
    assert not panel.close_button.isEnabled()
    assert not panel.advance_button.isEnabled()
    assert not panel.retreat_button.isEnabled()
    assert not panel.rc_button.isEnabled()


def test_opening_controls_follow_current_angle(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0)

    panel.show_opening(1.0)

    assert panel.opening_value.text() == "1.0°"
    assert panel.open_button.isEnabled()
    assert panel.close_button.isEnabled()
    assert panel.rc_button.isEnabled()

    panel.show_opening(30.0)

    assert not panel.open_button.isEnabled()
    assert panel.close_button.isEnabled()


def test_opening_buttons_emit_requests(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0)
    panel.show_opening(1.0)
    open_spy = QSignalSpy(panel.open_requested)
    close_spy = QSignalSpy(panel.close_requested)
    rc_spy = QSignalSpy(panel.rc_requested)

    panel.open_button.click()
    panel.close_button.click()
    panel.rc_button.click()

    assert open_spy.count() == 1
    assert close_spy.count() == 1
    assert rc_spy.count() == 1


def test_protrusion_controls_follow_current_distance(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0)

    panel.show_protrusion(1.0)

    assert panel.protrusion_value.text() == "1.0 mm"
    assert panel.advance_button.isEnabled()
    assert panel.retreat_button.isEnabled()
    assert panel.rc_button.isEnabled()

    panel.show_protrusion(17.0)

    assert not panel.advance_button.isEnabled()
    assert panel.retreat_button.isEnabled()


def test_protrusion_requires_known_path(qt_application) -> None:
    panel = FunctionalCalibrationPanel()

    panel.set_mounting_available(True)

    assert not panel.advance_button.isEnabled()
    assert not panel.retreat_button.isEnabled()


def test_protrusion_buttons_emit_requests(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0)
    panel.show_protrusion(1.0)
    advance_spy = QSignalSpy(panel.advance_requested)
    retreat_spy = QSignalSpy(panel.retreat_requested)

    panel.advance_button.click()
    panel.retreat_button.click()

    assert advance_spy.count() == 1
    assert retreat_spy.count() == 1
