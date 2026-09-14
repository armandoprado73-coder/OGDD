"""Tests for the protected RC-to-MIC diagnosis panel."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from ogdd.app.rc_mic_diagnosis_panel import RcMicDiagnosisPanel


@pytest.fixture(scope="module")
def qt_application():
    """Provide the one QApplication required by all panel tests."""

    return QApplication.instance() or QApplication([])


def registration_result(*, converged: bool = True):
    """Build the minimal registration interface displayed by the panel."""

    return SimpleNamespace(
        converged=converged,
        maxillary_registration=SimpleNamespace(
            root_mean_square_error=0.00000076,
        ),
        mandibular_registration=SimpleNamespace(
            root_mean_square_error=0.00202872,
        ),
    )


def test_panel_starts_protected(qt_application) -> None:
    panel = RcMicDiagnosisPanel()

    assert not panel.availability_label.isHidden()
    assert panel.maxillary_seed_value.text() == "Pendiente"
    assert panel.mandibular_seed_value.text() == "Pendiente"
    assert panel.convergence_value.text() == "Pendiente"
    assert not panel.maxillary_seed_button.isEnabled()
    assert not panel.mandibular_seed_button.isEnabled()
    assert not panel.diagnose_button.isEnabled()
    assert not panel.reset_button.isEnabled()
    assert not panel.view_rc_button.isEnabled()
    assert not panel.view_mic_button.isEnabled()
    assert not panel.view_overlay_button.isEnabled()


def test_record_alone_does_not_unlock_diagnosis(qt_application) -> None:
    panel = RcMicDiagnosisPanel()

    panel.set_prerequisites(mic_available=True, mounting_available=False)

    assert "montaje" in panel.availability_label.text()
    assert not panel.maxillary_seed_button.isEnabled()
    assert not panel.diagnose_button.isEnabled()


def test_mounting_without_record_reports_missing_mic(qt_application) -> None:
    panel = RcMicDiagnosisPanel()

    panel.set_prerequisites(mic_available=False, mounting_available=True)

    assert "registro MIC" in panel.availability_label.text()
    assert not panel.mandibular_seed_button.isEnabled()


def test_two_manual_seeds_unlock_diagnosis(qt_application) -> None:
    panel = RcMicDiagnosisPanel()
    panel.set_prerequisites(mic_available=True, mounting_available=True)

    panel.show_seed("maxillary", 28810, [-5.405, -27.969, 12.699])

    assert not panel.diagnose_button.isEnabled()

    panel.show_seed("mandibular", 74585, [18.811, 7.677, -14.881])

    assert panel.seeds_complete
    assert panel.diagnose_button.isEnabled()
    assert panel.reset_button.isEnabled()
    assert "Vértice 28,810" in panel.maxillary_seed_value.text()
    assert "Vértice 74,585" in panel.mandibular_seed_value.text()


def test_seed_buttons_emit_operator_requests(qt_application) -> None:
    panel = RcMicDiagnosisPanel()
    panel.set_prerequisites(mic_available=True, mounting_available=True)
    maxillary_spy = QSignalSpy(panel.select_maxillary_seed_requested)
    mandibular_spy = QSignalSpy(panel.select_mandibular_seed_requested)

    panel.maxillary_seed_button.click()
    panel.mandibular_seed_button.click()

    assert maxillary_spy.count() == 1
    assert mandibular_spy.count() == 1


def test_diagnose_request_requires_both_seeds(qt_application) -> None:
    panel = RcMicDiagnosisPanel()
    panel.set_prerequisites(mic_available=True, mounting_available=True)
    diagnose_spy = QSignalSpy(panel.diagnose_requested)

    panel.diagnose_button.click()
    panel.show_seed("maxillary", 1, [0.0, 0.0, 1.0])
    panel.diagnose_button.click()
    panel.show_seed("mandibular", 2, [0.0, 0.0, -1.0])
    panel.diagnose_button.click()

    assert diagnose_spy.count() == 1


def test_panel_reports_registration_and_local_vectors(qt_application) -> None:
    panel = RcMicDiagnosisPanel()
    panel.set_prerequisites(mic_available=True, mounting_available=True)
    panel.show_seed("maxillary", 1, [0.0, 0.0, 1.0])
    panel.show_seed("mandibular", 2, [0.0, 0.0, -1.0])

    panel.show_result(
        registration=registration_result(),
        right_vector=np.array([0.12345, -0.23456, 0.34567]),
        left_vector=np.array([-0.11111, 0.22222, -0.33333]),
    )

    assert panel.convergence_value.text() == "Convergente"
    assert panel.maxillary_rmse_value.text() == "0.000001 mm"
    assert panel.mandibular_rmse_value.text() == "0.002029 mm"
    assert panel.right_distance_value.text() == "0.4356 mm"
    assert panel.right_vector_value.text() == (
        "X +0.1235  |  Y -0.2346  |  Z +0.3457 mm"
    )
    assert panel.left_distance_value.text() == "0.4157 mm"
    assert panel.left_vector_value.text() == (
        "X -0.1111  |  Y +0.2222  |  Z -0.3333 mm"
    )
    assert "+X derecha" in panel.state_label.text()
    assert "+Y anterior" in panel.state_label.text()
    assert "+Z superior" in panel.state_label.text()
    assert panel.view_rc_button.isEnabled()
    assert panel.view_mic_button.isEnabled()
    assert panel.view_overlay_button.isEnabled()
    assert panel.diagnostic_view == "overlay"


def test_comparison_buttons_emit_exclusive_view_requests(qt_application) -> None:
    panel = RcMicDiagnosisPanel()
    panel.set_prerequisites(mic_available=True, mounting_available=True)
    panel.show_seed("maxillary", 1, [0.0, 0.0, 1.0])
    panel.show_seed("mandibular", 2, [0.0, 0.0, -1.0])
    panel.show_result(
        registration=registration_result(),
        right_vector=np.ones(3),
        left_vector=-np.ones(3),
    )
    rc_spy = QSignalSpy(panel.view_rc_requested)
    mic_spy = QSignalSpy(panel.view_mic_requested)
    overlay_spy = QSignalSpy(panel.view_overlay_requested)

    panel.view_rc_button.click()
    assert panel.diagnostic_view == "rc"
    panel.view_mic_button.click()
    assert panel.diagnostic_view == "mic"
    panel.view_overlay_button.click()
    assert panel.diagnostic_view == "overlay"

    assert rc_spy.count() == 1
    assert mic_spy.count() == 1
    assert overlay_spy.count() == 1


def test_reset_discards_seeds_and_results(qt_application) -> None:
    panel = RcMicDiagnosisPanel()
    panel.set_prerequisites(mic_available=True, mounting_available=True)
    panel.show_seed("maxillary", 1, [0.0, 0.0, 1.0])
    panel.show_seed("mandibular", 2, [0.0, 0.0, -1.0])
    panel.show_result(
        registration=registration_result(),
        right_vector=np.ones(3),
        left_vector=-np.ones(3),
    )

    panel.clear()

    assert not panel.seeds_complete
    assert panel.maxillary_seed_value.text() == "Pendiente"
    assert panel.mandibular_seed_value.text() == "Pendiente"
    assert panel.convergence_value.text() == "Pendiente"
    assert panel.right_distance_value.text() == "—"
    assert panel.left_distance_value.text() == "—"
    assert not panel.diagnose_button.isEnabled()
    assert not panel.reset_button.isEnabled()
