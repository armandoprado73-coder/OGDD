"""Tests for the functional calibration workflow panel."""

from __future__ import annotations

import os
from types import SimpleNamespace

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
    assert panel.active_path_value.text() == "RC"
    assert not panel.advance_button.isEnabled()
    assert not panel.retreat_button.isEnabled()
    assert not panel.left_button.isEnabled()
    assert not panel.right_button.isEnabled()
    assert not panel.lateral_zero_button.isEnabled()
    assert not panel.opening_slider.isEnabled()
    assert not panel.protrusion_slider.isEnabled()
    assert not panel.lateral_slider.isEnabled()
    assert not panel.fine_close_button.isEnabled()
    assert not panel.fine_open_button.isEnabled()
    assert not panel.adjustment_zero_button.isEnabled()
    assert not panel.save_protrusive_button.isEnabled()
    assert not panel.save_right_canine_button.isEnabled()
    assert not panel.save_left_canine_button.isEnabled()
    assert not panel.go_protrusive_button.isEnabled()
    assert not panel.go_right_canine_button.isEnabled()
    assert not panel.go_left_canine_button.isEnabled()
    assert not panel.clear_limits_button.isEnabled()


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
    assert panel.opening_slider.isEnabled()
    assert panel.protrusion_slider.isEnabled()
    assert not panel.lateral_slider.isEnabled()
    assert panel.fine_close_button.isEnabled()
    assert panel.fine_open_button.isEnabled()


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
    assert not panel.left_button.isEnabled()
    assert not panel.right_button.isEnabled()
    assert not panel.lateral_zero_button.isEnabled()
    assert not panel.rc_button.isEnabled()


def test_movement_buttons_report_fine_steps(qt_application) -> None:
    panel = FunctionalCalibrationPanel()

    assert panel.close_button.text() == "Cerrar 0.1°"
    assert panel.open_button.text() == "Abrir 0.1°"
    assert panel.retreat_button.text() == "Retroceder 0.1 mm"
    assert panel.advance_button.text() == "Avanzar 0.1 mm"
    assert panel.left_button.text() == "Izquierda 0.1°"
    assert panel.right_button.text() == "Derecha 0.1°"


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


def test_lateral_controls_follow_current_angle(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(
        True,
        17.0,
        30.0,
        3.0,
        3.0,
    )

    panel.show_lateral(-1.0)

    assert panel.lateral_value.text() == "-1.0° (izquierda)"
    assert panel.left_button.isEnabled()
    assert panel.right_button.isEnabled()
    assert panel.lateral_zero_button.isEnabled()
    assert panel.rc_button.isEnabled()

    panel.show_lateral(-3.0)

    assert not panel.left_button.isEnabled()
    assert panel.right_button.isEnabled()

    panel.show_lateral(3.0)

    assert panel.lateral_value.text() == "+3.0° (derecha)"
    assert panel.left_button.isEnabled()
    assert not panel.right_button.isEnabled()


def test_complete_position_updates_dynamic_limits(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 3.0, 3.0)

    panel.show_position(
        opening_angle_degrees=2.0,
        protrusion_distance_mm=10.0,
        lateral_angle_degrees=0.0,
        maximum_translation_mm=12.0,
        maximum_right_lateral_degrees=0.0,
        maximum_left_lateral_degrees=0.0,
    )

    assert panel.opening_value.text() == "2.0°"
    assert panel.protrusion_value.text() == "10.0 mm"
    assert panel.lateral_value.text() == "0.0°"
    assert panel.path_value.text() == "12.0 mm"
    assert panel.advance_button.isEnabled()
    assert not panel.left_button.isEnabled()
    assert not panel.right_button.isEnabled()


def test_lateral_buttons_emit_requests(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 3.0, 3.0)
    panel.show_lateral(1.0)
    left_spy = QSignalSpy(panel.move_left_requested)
    right_spy = QSignalSpy(panel.move_right_requested)
    zero_spy = QSignalSpy(panel.lateral_zero_requested)

    panel.left_button.click()
    panel.right_button.click()
    panel.lateral_zero_button.click()

    assert left_spy.count() == 1
    assert right_spy.count() == 1
    assert zero_spy.count() == 1


def test_sliders_emit_decimal_movement_requests(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 3.0, 4.0)
    opening_spy = QSignalSpy(panel.opening_changed)
    protrusion_spy = QSignalSpy(panel.protrusion_changed)
    lateral_spy = QSignalSpy(panel.lateral_changed)

    panel.opening_slider.setValue(14)
    panel.protrusion_slider.setValue(68)
    panel.lateral_slider.setValue(-23)

    assert opening_spy.count() == 1
    assert opening_spy.at(0)[0] == pytest.approx(1.4)
    assert protrusion_spy.count() == 1
    assert protrusion_spy.at(0)[0] == pytest.approx(6.8)
    assert lateral_spy.count() == 1
    assert lateral_spy.at(0)[0] == pytest.approx(-2.3)


def test_position_synchronizes_sliders_without_emitting(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 3.0, 4.0)
    opening_spy = QSignalSpy(panel.opening_changed)
    protrusion_spy = QSignalSpy(panel.protrusion_changed)
    lateral_spy = QSignalSpy(panel.lateral_changed)

    panel.show_position(
        opening_angle_degrees=1.4,
        protrusion_distance_mm=0.0,
        lateral_angle_degrees=-2.3,
        adjustment_angle_degrees=-0.6,
        maximum_translation_mm=17.0,
        maximum_right_lateral_degrees=3.0,
        maximum_left_lateral_degrees=4.0,
    )

    assert panel.opening_slider.value() == 14
    assert panel.protrusion_slider.value() == 0
    assert panel.lateral_slider.value() == -23
    assert panel.closure_value.text() == "-0.6° (cierre)"
    assert opening_spy.count() == 0
    assert protrusion_spy.count() == 0
    assert lateral_spy.count() == 0
    assert panel.adjustment_zero_button.isEnabled()
    assert panel.rc_button.isEnabled()


def test_fine_adjustment_buttons_emit_requests(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 3.0, 3.0)
    close_spy = QSignalSpy(panel.adjust_close_requested)
    open_spy = QSignalSpy(panel.adjust_open_requested)
    reset_spy = QSignalSpy(panel.adjustment_zero_requested)

    panel.fine_close_button.click()
    panel.fine_open_button.click()
    panel.show_adjustment(-0.1)
    panel.adjustment_zero_button.click()

    assert close_spy.count() == 1
    assert open_spy.count() == 1
    assert reset_spy.count() == 1


def make_limit(
    *,
    opening: float = 0.0,
    lateral: float = 0.0,
    protrusion: float = 0.0,
    adjustment: float = 0.0,
):
    """Build the numeric portion of one saved functional endpoint."""

    return SimpleNamespace(
        base_opening_angle_degrees=opening,
        lateral_angle_degrees=lateral,
        protrusion_distance_mm=protrusion,
        adjustment_angle_degrees=adjustment,
    )


def test_save_buttons_require_matching_clinical_positions(
    qt_application,
) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 4.0, 4.0)

    panel.show_protrusion(7.3)
    assert panel.save_protrusive_button.isEnabled()
    assert not panel.save_right_canine_button.isEnabled()
    assert not panel.save_left_canine_button.isEnabled()

    panel.show_protrusion(0.0)
    panel.show_lateral(2.3)
    assert not panel.save_protrusive_button.isEnabled()
    assert panel.save_right_canine_button.isEnabled()
    assert not panel.save_left_canine_button.isEnabled()

    panel.show_lateral(-3.4)
    assert not panel.save_protrusive_button.isEnabled()
    assert not panel.save_right_canine_button.isEnabled()
    assert panel.save_left_canine_button.isEnabled()


def test_active_path_locks_incompatible_movement_controls(
    qt_application,
) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 4.0, 4.0)

    panel.show_protrusion(7.3)
    assert panel.active_path_value.text() == "Protrusiva desde RC"
    assert panel.protrusion_slider.isEnabled()
    assert not panel.lateral_slider.isEnabled()
    assert not panel.left_button.isEnabled()
    assert not panel.right_button.isEnabled()

    panel.show_protrusion(0.0)
    panel.show_lateral(2.3)
    assert panel.active_path_value.text() == "Canina derecha desde RC"
    assert not panel.protrusion_slider.isEnabled()
    assert panel.lateral_slider.isEnabled()
    assert panel.lateral_slider.minimum() == 0
    assert panel.lateral_slider.maximum() == 40


def test_hinge_movement_requires_rc_before_excursion(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 4.0, 4.0)

    panel.show_opening(1.2)

    assert "vuelva a RC" in panel.active_path_value.text()
    assert not panel.protrusion_slider.isEnabled()
    assert not panel.lateral_slider.isEnabled()


def test_saved_limits_show_complete_numeric_calibration(
    qt_application,
) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 4.0, 4.0)

    panel.show_limits(
        protrusive_limit=make_limit(protrusion=7.3, adjustment=-0.9),
        right_canine_limit=make_limit(opening=1.1, lateral=2.3),
        left_canine_limit=make_limit(opening=1.1, lateral=-3.4),
    )

    assert panel.protrusive_limit_value.text() == (
        "A 0.0° | P 7.3 mm | L +0.0° | ajuste -0.9°"
    )
    assert panel.right_canine_limit_value.text() == (
        "A 1.1° | P 0.0 mm | L +2.3° | ajuste +0.0°"
    )
    assert panel.left_canine_limit_value.text() == (
        "A 1.1° | P 0.0 mm | L -3.4° | ajuste +0.0°"
    )
    assert panel.go_protrusive_button.isEnabled()
    assert panel.go_right_canine_button.isEnabled()
    assert panel.go_left_canine_button.isEnabled()
    assert panel.clear_limits_button.isEnabled()
    assert not panel.save_protrusive_button.isEnabled()
    assert not panel.save_right_canine_button.isEnabled()
    assert not panel.save_left_canine_button.isEnabled()


def test_limit_buttons_emit_operator_requests(qt_application) -> None:
    panel = FunctionalCalibrationPanel()
    panel.set_mounting_available(True, 17.0, 30.0, 4.0, 4.0)
    panel.show_protrusion(7.3)
    save_protrusive_spy = QSignalSpy(panel.save_protrusive_requested)
    save_right_spy = QSignalSpy(panel.save_right_canine_requested)
    save_left_spy = QSignalSpy(panel.save_left_canine_requested)
    go_protrusive_spy = QSignalSpy(panel.go_protrusive_requested)
    go_right_spy = QSignalSpy(panel.go_right_canine_requested)
    go_left_spy = QSignalSpy(panel.go_left_canine_requested)
    clear_spy = QSignalSpy(panel.clear_limits_requested)

    panel.save_protrusive_button.click()
    panel.show_limits(
        protrusive_limit=make_limit(protrusion=7.3),
        right_canine_limit=make_limit(lateral=2.3),
        left_canine_limit=make_limit(lateral=-3.4),
    )
    panel.go_protrusive_button.click()
    panel.go_right_canine_button.click()
    panel.go_left_canine_button.click()
    panel.clear_limits_button.click()

    assert save_protrusive_spy.count() == 1
    assert save_right_spy.count() == 0
    assert save_left_spy.count() == 0
    assert go_protrusive_spy.count() == 1
    assert go_right_spy.count() == 1
    assert go_left_spy.count() == 1
    assert clear_spy.count() == 1
