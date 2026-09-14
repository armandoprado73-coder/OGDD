"""Tests for point-7 clinical result exports."""

from __future__ import annotations

import csv
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage
from PySide6.QtPdf import QPdfDocument
from PySide6.QtWidgets import QApplication

from ogdd.app.results_export import ResultsExporter, StudyResultsSnapshot


@pytest.fixture(scope="module")
def qt_application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def snapshot() -> StudyResultsSnapshot:
    return StudyResultsSnapshot(
        study_name="Paciente de prueba",
        source_files={
            "maxillary_rc": "maxilar.stl",
            "mandibular_rc": "mandibula.stl",
            "mic_record": "registro.stl",
        },
        mounting={
            "intercondylar_width": 110.0,
            "balkwill_angle_degrees": 25.0,
            "right_condylar_guidance_degrees": 45.0,
            "left_condylar_guidance_degrees": 46.0,
            "functional_path_mm": 17.0,
        },
        functional_limits=(
            {
                "kind": "protrusive_edge_to_edge",
                "base_opening_angle_degrees": 0.2,
                "adjustment_angle_degrees": -0.1,
                "total_opening_angle_degrees": 0.1,
                "lateral_angle_degrees": 0.0,
                "protrusion_distance_mm": 7.3,
            },
        ),
        rc_mic={
            "converged": True,
            "maxillary_rmse_mm": 0.00000076,
            "mandibular_rmse_mm": 0.00202872,
            "right_vector_mm": [0.1, -0.2, 0.3],
            "right_distance_mm": 0.37416574,
            "left_vector_mm": [-0.4, 0.5, -0.6],
            "left_distance_mm": 0.87749644,
        },
        generated_at_utc="2026-09-14T04:00:00+00:00",
    )


def test_csv_rows_have_stable_anatomical_axes(snapshot) -> None:
    rows = ResultsExporter.csv_rows(snapshot)

    assert ("axes", "positive_x", "patient right", "") in rows
    assert ("axes", "positive_y", "anterior", "") in rows
    assert ("axes", "positive_z", "superior", "") in rows


def test_csv_export_preserves_numeric_result_values(tmp_path, snapshot) -> None:
    path = tmp_path / "results.csv"

    ResultsExporter.export_csv(path, snapshot)

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    right_x = next(row for row in rows if row["metric"] == "right_x")
    mandibular_rmse = next(
        row for row in rows if row["metric"] == "mandibular_rmse"
    )
    assert right_x == {
        "section": "rc_mic",
        "metric": "right_x",
        "value": "0.1",
        "unit": "mm",
    }
    assert mandibular_rmse["value"] == "0.00202872"


def test_pdf_html_escapes_study_name_and_lists_diagnosis(snapshot) -> None:
    altered = StudyResultsSnapshot(
        **{
            **snapshot.__dict__,
            "study_name": "Paciente <A&B>",
        }
    )

    html = ResultsExporter.pdf_html(altered)

    assert "Paciente &lt;A&amp;B&gt;" in html
    assert "Diagnóstico RC - MIC" in html
    assert "X +0.1000 mm" in html
    assert "RMSE mandibular" in html


def test_pdf_export_creates_valid_pdf_header(
    tmp_path,
    snapshot,
    qt_application,
) -> None:
    path = tmp_path / "results.pdf"

    ResultsExporter.export_pdf(path, snapshot)

    assert path.read_bytes().startswith(b"%PDF-")
    assert path.stat().st_size > 5_000


def test_visual_page_has_six_clinical_positions(snapshot) -> None:
    image_urls = {
        "overlay_right": "ogdd-image:///overlay_right",
        "overlay_front": "ogdd-image:///overlay_front",
        "overlay_left": "ogdd-image:///overlay_left",
        "right_canine": None,
        "protrusive": "ogdd-image:///protrusive",
        "left_canine": None,
    }

    html = ResultsExporter.pdf_html(snapshot, image_urls=image_urls)

    assert "Registro visual del montaje" in html
    assert "Vista derecha" in html
    assert "Vista frontal" in html
    assert "Vista izquierda" in html
    assert "Canina derecha" in html
    assert "Borde a borde" in html
    assert "Canina izquierda" in html
    assert html.count("Posición no registrada") >= 2


def test_pdf_with_visual_captures_has_two_pages(
    tmp_path,
    snapshot,
    qt_application,
) -> None:
    capture = tmp_path / "capture.png"
    image = QImage(960, 720, QImage.Format.Format_RGB32)
    image.fill(QColor("#263944"))
    assert image.save(str(capture))
    captures = {
        "overlay_right": capture,
        "overlay_front": capture,
        "overlay_left": capture,
        "right_canine": None,
        "protrusive": capture,
        "left_canine": None,
    }
    path = tmp_path / "visual-results.pdf"

    ResultsExporter.export_pdf(
        path,
        snapshot,
        visual_captures=captures,
    )

    document = QPdfDocument()
    assert document.load(str(path)) == QPdfDocument.Error.None_
    assert document.pageCount() == 2
