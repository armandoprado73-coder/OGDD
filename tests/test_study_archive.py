"""Tests for the portable native OGDD study archive."""

from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from ogdd.app.study_archive import StudyArchive
from ogdd.mesh import Mesh


def triangle_mesh(offset: float = 0.0) -> Mesh:
    return Mesh(
        vertices=np.array(
            [
                [offset, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        normals=np.array([[0.0, 0.0, 1.0]]),
        attributes={"region": np.array([1, 1, 1])},
        metadata={"arch": "test", "scale": np.float64(1.0)},
    )


def study_meshes() -> dict[str, Mesh]:
    return {
        "maxillary_rc": triangle_mesh(0.0),
        "mandibular_rc": triangle_mesh(2.0),
        "mic_record": triangle_mesh(4.0),
    }


def test_round_trip_preserves_meshes_and_clinical_state(tmp_path) -> None:
    path = tmp_path / "patient.ogdd"
    state = {
        "landmarks": {"DENTAL_MIDLINE": [0.0, 1.0, 2.0]},
        "mounting": {"intercondylar_width": 110.0},
        "mic_seed_vertices": {"maxillary": 1, "mandibular": 2},
    }

    StudyArchive.save(
        path,
        meshes=study_meshes(),
        filenames={
            "maxillary_rc": "Maxillary Anatomy.stl",
            "mandibular_rc": "Mandibular Anatomy.stl",
            "mic_record": "registro.stl",
        },
        state=state,
    )
    loaded = StudyArchive.load(path)

    assert loaded.state == state
    assert loaded.filenames["mic_record"] == "registro.stl"
    assert set(loaded.meshes) == {
        "maxillary_rc",
        "mandibular_rc",
        "mic_record",
    }
    for name, original in study_meshes().items():
        restored = loaded.meshes[name]
        assert np.array_equal(restored.vertices, original.vertices)
        assert np.array_equal(restored.faces, original.faces)
        assert np.array_equal(restored.normals, original.normals)
        assert np.array_equal(
            restored.attributes["region"],
            original.attributes["region"],
        )
        assert restored.metadata == {"arch": "test", "scale": 1.0}


def test_archive_contains_manifest_and_internal_arrays(tmp_path) -> None:
    path = tmp_path / "patient.ogdd"
    StudyArchive.save(
        path,
        meshes=study_meshes(),
        filenames={},
        state={},
    )

    with ZipFile(path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))

    assert "meshes/maxillary_rc/vertices.npy" in names
    assert "meshes/mandibular_rc/faces.npy" in names
    assert manifest["format"] == "OGDD_STUDY"
    assert manifest["format_version"] == 1


def test_save_is_atomic_when_state_is_not_serializable(tmp_path) -> None:
    path = tmp_path / "patient.ogdd"
    path.write_bytes(b"original")

    with pytest.raises(TypeError, match="JSON-compatible"):
        StudyArchive.save(
            path,
            meshes=study_meshes(),
            filenames={},
            state={"invalid": object()},
        )

    assert path.read_bytes() == b"original"


def test_missing_required_mesh_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="mandibular_rc"):
        StudyArchive.save(
            tmp_path / "patient.ogdd",
            meshes={"maxillary_rc": triangle_mesh()},
            filenames={},
            state={},
        )


def test_non_archive_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "patient.ogdd"
    path.write_text("not a zip", encoding="utf-8")

    with pytest.raises(ValueError, match="valid OGDD"):
        StudyArchive.load(path)


def test_unsupported_version_is_rejected(tmp_path) -> None:
    path = tmp_path / "future.ogdd"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "OGDD_STUDY",
                    "format_version": 99,
                    "meshes": {},
                    "state": {},
                }
            ),
        )

    with pytest.raises(ValueError, match="version"):
        StudyArchive.load(path)


def test_modified_array_fails_integrity_check(tmp_path) -> None:
    path = tmp_path / "patient.ogdd"
    StudyArchive.save(
        path,
        meshes=study_meshes(),
        filenames={},
        state={},
    )
    with ZipFile(path) as source:
        manifest = json.loads(source.read("manifest.json"))
        members = {
            name: source.read(name)
            for name in source.namelist()
            if name
            not in {
                "manifest.json",
                "meshes/maxillary_rc/vertices.npy",
            }
        }

    stream = BytesIO()
    np.save(stream, np.zeros((3, 3)), allow_pickle=False)
    members["meshes/maxillary_rc/vertices.npy"] = stream.getvalue()
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as target:
        for name, data in members.items():
            target.writestr(name, data)
        target.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(ValueError, match="integrity"):
        StudyArchive.load(path)
