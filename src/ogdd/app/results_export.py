"""Clinical result snapshots and portable exports for OGDD studies."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from PySide6.QtCore import QMarginsF, QUrl
from PySide6.QtGui import (
    QImage,
    QPageLayout,
    QPageSize,
    QPdfWriter,
    QTextDocument,
)


@dataclass(frozen=True)
class StudyResultsSnapshot:
    """Serializable clinical summary used by every point-7 export."""

    study_name: str
    source_files: dict[str, str] = field(default_factory=dict)
    mounting: dict[str, Any] | None = None
    functional_limits: tuple[dict[str, Any], ...] = ()
    rc_mic: dict[str, Any] | None = None
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ResultsExporter:
    """Write clinical reports without changing the study state."""

    @staticmethod
    def export_csv(path: str | Path, snapshot: StudyResultsSnapshot) -> Path:
        """Write long-form UTF-8 clinical values suitable for analysis."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["section", "metric", "value", "unit"])
            for row in ResultsExporter.csv_rows(snapshot):
                writer.writerow(row)
        return path

    @staticmethod
    def csv_rows(snapshot: StudyResultsSnapshot) -> list[tuple[str, str, Any, str]]:
        """Return stable, auditable output rows for one result snapshot."""

        rows: list[tuple[str, str, Any, str]] = [
            ("study", "name", snapshot.study_name, ""),
            ("study", "generated_at_utc", snapshot.generated_at_utc, ""),
            ("axes", "positive_x", "patient right", ""),
            ("axes", "positive_y", "anterior", ""),
            ("axes", "positive_z", "superior", ""),
        ]
        for key, value in sorted(snapshot.source_files.items()):
            rows.append(("source_files", key, value, ""))

        mounting = snapshot.mounting
        if mounting is not None:
            rows.extend(
                [
                    (
                        "mounting",
                        "intercondylar_width",
                        mounting["intercondylar_width"],
                        "mm",
                    ),
                    (
                        "mounting",
                        "balkwill_angle",
                        mounting["balkwill_angle_degrees"],
                        "degrees",
                    ),
                    (
                        "mounting",
                        "right_condylar_guidance",
                        mounting["right_condylar_guidance_degrees"],
                        "degrees",
                    ),
                    (
                        "mounting",
                        "left_condylar_guidance",
                        mounting["left_condylar_guidance_degrees"],
                        "degrees",
                    ),
                    (
                        "mounting",
                        "functional_path",
                        mounting["functional_path_mm"],
                        "mm",
                    ),
                ]
            )

        for limit in snapshot.functional_limits:
            section = f"functional_limit:{limit['kind']}"
            rows.extend(
                [
                    (
                        section,
                        "base_opening",
                        limit["base_opening_angle_degrees"],
                        "degrees",
                    ),
                    (
                        section,
                        "occlusal_adjustment",
                        limit["adjustment_angle_degrees"],
                        "degrees",
                    ),
                    (
                        section,
                        "total_opening",
                        limit["total_opening_angle_degrees"],
                        "degrees",
                    ),
                    (
                        section,
                        "lateral",
                        limit["lateral_angle_degrees"],
                        "degrees",
                    ),
                    (
                        section,
                        "protrusion",
                        limit["protrusion_distance_mm"],
                        "mm",
                    ),
                ]
            )

        diagnosis = snapshot.rc_mic
        if diagnosis is not None:
            rows.extend(
                [
                    ("rc_mic", "converged", diagnosis["converged"], ""),
                    (
                        "rc_mic",
                        "maxillary_rmse",
                        diagnosis["maxillary_rmse_mm"],
                        "mm",
                    ),
                    (
                        "rc_mic",
                        "mandibular_rmse",
                        diagnosis["mandibular_rmse_mm"],
                        "mm",
                    ),
                ]
            )
            for side in ("right", "left"):
                vector = diagnosis[f"{side}_vector_mm"]
                rows.extend(
                    [
                        ("rc_mic", f"{side}_x", vector[0], "mm"),
                        ("rc_mic", f"{side}_y", vector[1], "mm"),
                        ("rc_mic", f"{side}_z", vector[2], "mm"),
                        (
                            "rc_mic",
                            f"{side}_distance",
                            diagnosis[f"{side}_distance_mm"],
                            "mm",
                        ),
                    ]
                )
        return rows

    @staticmethod
    def export_pdf(
        path: str | Path,
        snapshot: StudyResultsSnapshot,
        *,
        visual_captures: dict[str, Path | None] | None = None,
    ) -> Path:
        """Write an A4 clinical report through the existing Qt runtime."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = QTextDocument()
        document.setDocumentMargin(0)
        image_urls: dict[str, str | None] | None = None
        if visual_captures is not None:
            image_urls = {}
            for key, capture_path in visual_captures.items():
                if capture_path is None:
                    image_urls[key] = None
                    continue
                image = QImage(str(capture_path))
                if image.isNull():
                    raise OSError(
                        f"OGDD could not read report capture: {capture_path}"
                    )
                url = QUrl(f"ogdd-image:///{key}")
                document.addResource(
                    QTextDocument.ResourceType.ImageResource,
                    url,
                    image,
                )
                image_urls[key] = url.toString()
        document.setHtml(
            ResultsExporter.pdf_html(snapshot, image_urls=image_urls)
        )

        writer = QPdfWriter(str(path))
        writer.setResolution(300)
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        writer.setPageMargins(
            QMarginsF(14.0, 14.0, 14.0, 14.0),
            QPageLayout.Unit.Millimeter,
        )
        document.print_(writer)
        del writer
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError("OGDD could not create the PDF report.")
        return path

    @staticmethod
    def pdf_html(
        snapshot: StudyResultsSnapshot,
        *,
        image_urls: dict[str, str | None] | None = None,
    ) -> str:
        """Build deterministic, escaped HTML used by the PDF printer."""

        sections = [
            ResultsExporter._html_table(
                "Archivos del estudio",
                [
                    (ResultsExporter._source_label(key), value)
                    for key, value in sorted(snapshot.source_files.items())
                ],
            )
        ]
        if snapshot.mounting is not None:
            mounting = snapshot.mounting
            sections.append(
                ResultsExporter._html_table(
                    "Montaje virtual en RC",
                    [
                        (
                            "Distancia intercondilar",
                            f"{mounting['intercondylar_width']:.1f} mm",
                        ),
                        (
                            "Ángulo de Balkwill",
                            f"{mounting['balkwill_angle_degrees']:.1f} grados",
                        ),
                        (
                            "Guía condilar derecha",
                            f"{mounting['right_condylar_guidance_degrees']:.1f} grados",
                        ),
                        (
                            "Guía condilar izquierda",
                            f"{mounting['left_condylar_guidance_degrees']:.1f} grados",
                        ),
                        (
                            "Recorrido funcional",
                            f"{mounting['functional_path_mm']:.1f} mm",
                        ),
                    ],
                )
            )

        if snapshot.functional_limits:
            limit_rows = []
            labels = {
                "protrusive_edge_to_edge": "Borde a borde protrusivo",
                "right_canine_cusp_to_cusp": "Protección canina derecha",
                "left_canine_cusp_to_cusp": "Protección canina izquierda",
            }
            for limit in snapshot.functional_limits:
                value = (
                    f"A {limit['base_opening_angle_degrees']:.1f} grados, "
                    f"P {limit['protrusion_distance_mm']:.1f} mm, "
                    f"L {limit['lateral_angle_degrees']:+.1f} grados, "
                    f"ajuste {limit['adjustment_angle_degrees']:+.1f} grados"
                )
                limit_rows.append((labels[limit["kind"]], value))
            sections.append(
                ResultsExporter._html_table(
                    "Calibración funcional",
                    limit_rows,
                )
            )

        if snapshot.rc_mic is not None:
            diagnosis = snapshot.rc_mic
            right = diagnosis["right_vector_mm"]
            left = diagnosis["left_vector_mm"]
            sections.append(
                ResultsExporter._html_table(
                    "Diagnóstico RC - MIC",
                    [
                        (
                            "Registro",
                            "Convergente"
                            if diagnosis["converged"]
                            else "No convergente",
                        ),
                        (
                            "RMSE maxilar",
                            f"{diagnosis['maxillary_rmse_mm']:.6f} mm",
                        ),
                        (
                            "RMSE mandibular",
                            f"{diagnosis['mandibular_rmse_mm']:.6f} mm",
                        ),
                        (
                            "Cóndilo derecho",
                            ResultsExporter._vector_text(
                                right,
                                diagnosis["right_distance_mm"],
                            ),
                        ),
                        (
                            "Cóndilo izquierdo",
                            ResultsExporter._vector_text(
                                left,
                                diagnosis["left_distance_mm"],
                            ),
                        ),
                    ],
                )
            )

        visual_page = ResultsExporter._visual_page(snapshot, image_urls)
        return f"""
        <html>
        <head>
        <style>
            body {{ font-family: 'Arial'; color: #20323a; font-size: 10pt; }}
            h1 {{ color: #24484b; font-size: 20pt; margin-bottom: 2px; }}
            h2 {{ color: #315b5f; font-size: 12pt; margin-top: 18px;
                  margin-bottom: 6px; }}
            .subtitle {{ color: #60757d; margin-bottom: 14px; }}
            .axes {{ background: #e8f1f2; border: 1px solid #c9dadd;
                     padding: 8px; margin: 10px 0 12px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            td {{ border-bottom: 1px solid #d9e2e5; padding: 5px; }}
            td:first-child {{ width: 38%; font-weight: bold; color: #315b5f; }}
            .page-break {{ page-break-before: always; }}
            .visual-grid {{ width: 100%; border-collapse: separate;
                            border-spacing: 3px; table-layout: fixed; }}
            .visual-grid td {{ width: 33%; padding: 6px; text-align: center;
                               vertical-align: top; border: 1px solid #d9e2e5; }}
            .visual-grid td:first-child {{ width: 33%; font-weight: normal;
                                           color: #20323a; }}
            .visual-title {{ font-weight: bold; color: #315b5f;
                             font-size: 9pt; margin-bottom: 4px; }}
            .visual-detail {{ color: #60757d; font-size: 8pt;
                              margin-top: 4px; }}
            .visual-missing {{ background: #f3f6f7; color: #71838a;
                               padding: 58px 4px; }}
            .footer {{ color: #71838a; font-size: 8pt; margin-top: 20px; }}
        </style>
        </head>
        <body>
            <h1>OGDD</h1>
            <div class="subtitle">Open Geometry for Digital Dentistry<br>
            Informe de resultados: {escape(snapshot.study_name)}</div>
            <div class="axes"><b>Ejes anatómicos:</b> +X derecha del paciente,
            +Y anterior, +Z superior.</div>
            {''.join(sections)}
            <div class="footer">Generado en UTC: {escape(snapshot.generated_at_utc)}<br>
            El informe conserva los valores geométricos calculados por OGDD.</div>
            {visual_page}
        </body>
        </html>
        """

    @staticmethod
    def _html_table(title: str, rows: list[tuple[str, Any]]) -> str:
        if not rows:
            return ""
        body = "".join(
            f"<tr><td>{escape(str(label))}</td>"
            f"<td>{escape(str(value))}</td></tr>"
            for label, value in rows
        )
        return f"<h2>{escape(title)}</h2><table>{body}</table>"

    @staticmethod
    def _visual_page(
        snapshot: StudyResultsSnapshot,
        image_urls: dict[str, str | None] | None,
    ) -> str:
        """Build the standardized second report page when requested."""

        if image_urls is None:
            return ""
        limits = {
            limit["kind"]: limit for limit in snapshot.functional_limits
        }
        cards = (
            (
                "overlay_right",
                "Vista derecha",
                "Superposición RC/MIC",
            ),
            (
                "overlay_front",
                "Vista frontal",
                "Superposición RC/MIC",
            ),
            (
                "overlay_left",
                "Vista izquierda",
                "Superposición RC/MIC",
            ),
            (
                "right_canine",
                "Canina derecha",
                ResultsExporter._limit_caption(
                    limits.get("right_canine_cusp_to_cusp")
                ),
            ),
            (
                "protrusive",
                "Borde a borde",
                ResultsExporter._limit_caption(
                    limits.get("protrusive_edge_to_edge")
                ),
            ),
            (
                "left_canine",
                "Canina izquierda",
                ResultsExporter._limit_caption(
                    limits.get("left_canine_cusp_to_cusp")
                ),
            ),
        )
        cells = [
            ResultsExporter._visual_card(
                title,
                detail,
                image_urls.get(key),
            )
            for key, title, detail in cards
        ]
        rows = "".join(
            "<tr>" + "".join(cells[index : index + 3]) + "</tr>"
            for index in range(0, len(cells), 3)
        )
        return f"""
        <div class="page-break"></div>
        <h1>Registro visual del montaje</h1>
        <div class="subtitle">{escape(snapshot.study_name)}<br>
        Comparación diagnóstica y posiciones funcionales confirmadas.</div>
        <table class="visual-grid">{rows}</table>
        <div class="footer">Las vistas fueron generadas automáticamente con
        la orientación anatómica y los límites guardados en el estudio.</div>
        """

    @staticmethod
    def _visual_card(
        title: str,
        detail: str,
        image_url: str | None,
    ) -> str:
        if image_url is None:
            image = '<div class="visual-missing">Posición no registrada</div>'
        else:
            image = (
                f'<img src="{escape(image_url)}" width="180" height="135">'
            )
        return (
            f'<td><div class="visual-title">{escape(title)}</div>'
            f'{image}<div class="visual-detail">{escape(detail)}</div></td>'
        )

    @staticmethod
    def _limit_caption(limit: dict[str, Any] | None) -> str:
        if limit is None:
            return "Posición no registrada"
        return (
            f"A {limit['total_opening_angle_degrees']:.1f}°, "
            f"P {limit['protrusion_distance_mm']:.1f} mm, "
            f"L {limit['lateral_angle_degrees']:+.1f}°"
        )

    @staticmethod
    def _source_label(key: str) -> str:
        return {
            "maxillary_rc": "Maxilar en RC",
            "mandibular_rc": "Mandíbula en RC",
            "mic_record": "Registro MIC",
        }.get(key, key)

    @staticmethod
    def _vector_text(vector: list[float], distance: float) -> str:
        return (
            f"X {vector[0]:+.4f} mm, Y {vector[1]:+.4f} mm, "
            f"Z {vector[2]:+.4f} mm; magnitud {distance:.4f} mm"
        )
