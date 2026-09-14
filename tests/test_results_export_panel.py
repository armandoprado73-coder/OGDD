"""Tests for the point-7 results and export panel."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from ogdd.app.results_export import StudyResultsSnapshot
from ogdd.app.results_export_panel import ResultsExportPanel


@pytest.fixture(scope="module")
def qt_application():
    return QApplication.instance() or QApplication([])


def complete_snapshot() -> StudyResultsSnapshot:
    return StudyResultsSnapshot(
        study_name="estudio_rc_mic",
        mounting={"intercondylar_width": 110.0},
        functional_limits=({}, {}, {}),
        rc_mic={"converged": True},
    )


def test_panel_starts_protected(qt_application) -> None:
    panel = ResultsExportPanel()

    assert not panel.availability_label.isHidden()
    assert not panel.save_study_button.isEnabled()
    assert not panel.export_pdf_button.isEnabled()
    assert not panel.export_csv_button.isEnabled()
    assert not panel.export_png_button.isEnabled()


def test_loaded_models_enable_native_save_and_png(qt_application) -> None:
    panel = ResultsExportPanel()
    panel.set_study_available(True)
    panel.show_snapshot(StudyResultsSnapshot(study_name="estudio"))

    assert panel.save_study_button.isEnabled()
    assert panel.export_png_button.isEnabled()
    assert not panel.export_pdf_button.isEnabled()
    assert not panel.export_csv_button.isEnabled()


def test_clinical_results_enable_pdf_and_csv(qt_application) -> None:
    panel = ResultsExportPanel()

    panel.show_snapshot(complete_snapshot())

    assert panel.study_value.text() == "estudio_rc_mic"
    assert panel.mounting_value.text() == "Construido"
    assert panel.limits_value.text() == "3 de 3"
    assert panel.diagnosis_value.text() == "Convergente"
    assert panel.export_pdf_button.isEnabled()
    assert panel.export_csv_button.isEnabled()


def test_output_buttons_emit_explicit_requests(qt_application) -> None:
    panel = ResultsExportPanel()
    panel.show_snapshot(complete_snapshot())
    save_spy = QSignalSpy(panel.save_study_requested)
    pdf_spy = QSignalSpy(panel.export_pdf_requested)
    csv_spy = QSignalSpy(panel.export_csv_requested)
    png_spy = QSignalSpy(panel.export_png_requested)

    panel.save_study_button.click()
    panel.export_pdf_button.click()
    panel.export_csv_button.click()
    panel.export_png_button.click()

    assert save_spy.count() == 1
    assert pdf_spy.count() == 1
    assert csv_spy.count() == 1
    assert png_spy.count() == 1
