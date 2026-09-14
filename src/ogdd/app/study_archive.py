"""Portable, versioned storage for complete OGDD studies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

import numpy as np

from ogdd.mesh import Mesh


@dataclass(frozen=True)
class StudyArchiveData:
    """Meshes, display filenames and clinical state loaded from one archive."""

    meshes: dict[str, Mesh]
    filenames: dict[str, str]
    state: dict[str, Any]


class StudyArchive:
    """Read and write the native single-file ``.ogdd`` study format."""

    FORMAT_NAME = "OGDD_STUDY"
    FORMAT_VERSION = 1
    MAX_MEMBER_BYTES = 2_000_000_000
    REQUIRED_MESHES = frozenset({"maxillary_rc", "mandibular_rc"})

    @classmethod
    def save(
        cls,
        path: str | Path,
        *,
        meshes: dict[str, Mesh],
        filenames: dict[str, str],
        state: dict[str, Any],
    ) -> Path:
        """Atomically save one self-contained native OGDD study."""

        path = Path(path)
        cls._validate_meshes(meshes)
        serialized_state = cls._json_value(state, "Clinical state")
        serialized_filenames = {
            str(key): Path(str(value)).name
            for key, value in filenames.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            with ZipFile(
                temporary_path,
                mode="w",
                compression=ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                mesh_entries = {
                    name: cls._write_mesh(archive, name, mesh)
                    for name, mesh in sorted(meshes.items())
                }
                manifest = {
                    "format": cls.FORMAT_NAME,
                    "format_version": cls.FORMAT_VERSION,
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "filenames": serialized_filenames,
                    "meshes": mesh_entries,
                    "state": serialized_state,
                }
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    ).encode("utf-8"),
                )
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> StudyArchiveData:
        """Load and validate one native OGDD study without extracting files."""

        path = Path(path)
        try:
            with ZipFile(path, mode="r") as archive:
                cls._validate_archive_members(archive)
                manifest = cls._read_manifest(archive)
                mesh_descriptors = manifest.get("meshes")
                if not isinstance(mesh_descriptors, dict):
                    raise ValueError("OGDD manifest has no mesh collection.")
                meshes = {
                    name: cls._read_mesh(archive, name, descriptor)
                    for name, descriptor in mesh_descriptors.items()
                }
        except BadZipFile as error:
            raise ValueError("File is not a valid OGDD study archive.") from error

        cls._validate_meshes(meshes)
        filenames = manifest.get("filenames", {})
        state = manifest.get("state", {})
        if not isinstance(filenames, dict):
            raise ValueError("OGDD filenames must be a mapping.")
        if not isinstance(state, dict):
            raise ValueError("OGDD clinical state must be a mapping.")
        return StudyArchiveData(
            meshes=meshes,
            filenames={
                str(key): Path(str(value)).name
                for key, value in filenames.items()
            },
            state=state,
        )

    @classmethod
    def _write_mesh(
        cls,
        archive: ZipFile,
        name: str,
        mesh: Mesh,
    ) -> dict[str, Any]:
        base = f"meshes/{name}"
        attributes = {
            attribute_name: cls._write_array(
                archive,
                f"{base}/attributes/{index}.npy",
                values,
            )
            for index, (attribute_name, values) in enumerate(
                sorted(mesh.attributes.items())
            )
        }
        return {
            "vertices": cls._write_array(
                archive,
                f"{base}/vertices.npy",
                mesh.vertices,
            ),
            "faces": cls._write_array(
                archive,
                f"{base}/faces.npy",
                mesh.faces,
            ),
            "normals": (
                None
                if mesh.normals is None
                else cls._write_array(
                    archive,
                    f"{base}/normals.npy",
                    mesh.normals,
                )
            ),
            "attributes": attributes,
            "metadata": cls._json_value(
                mesh.metadata,
                f"Metadata for mesh {name}",
            ),
        }

    @classmethod
    def _read_mesh(
        cls,
        archive: ZipFile,
        name: str,
        descriptor: Any,
    ) -> Mesh:
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError("OGDD mesh descriptor is invalid.")
        attributes = descriptor.get("attributes", {})
        if not isinstance(attributes, dict):
            raise ValueError(f"Attributes for mesh {name} are invalid.")
        normals_descriptor = descriptor.get("normals")
        return Mesh(
            vertices=cls._read_array(
                archive,
                descriptor.get("vertices"),
            ),
            faces=cls._read_array(
                archive,
                descriptor.get("faces"),
            ),
            normals=(
                None
                if normals_descriptor is None
                else cls._read_array(archive, normals_descriptor)
            ),
            attributes={
                str(attribute_name): cls._read_array(
                    archive,
                    attribute_descriptor,
                )
                for attribute_name, attribute_descriptor
                in attributes.items()
            },
            metadata=descriptor.get("metadata", {}),
        )

    @staticmethod
    def _array_bytes(values: np.ndarray) -> bytes:
        stream = BytesIO()
        np.save(stream, np.asarray(values), allow_pickle=False)
        return stream.getvalue()

    @classmethod
    def _write_array(
        cls,
        archive: ZipFile,
        member_name: str,
        values: np.ndarray,
    ) -> dict[str, Any]:
        data = cls._array_bytes(values)
        archive.writestr(member_name, data)
        return {
            "path": member_name,
            "size": len(data),
            "sha256": sha256(data).hexdigest(),
        }

    @classmethod
    def _read_array(cls, archive: ZipFile, descriptor: Any) -> np.ndarray:
        if not isinstance(descriptor, dict):
            raise ValueError("OGDD array descriptor is invalid.")
        member_name = descriptor.get("path")
        expected_size = descriptor.get("size")
        expected_digest = descriptor.get("sha256")
        if (
            not isinstance(member_name, str)
            or not isinstance(expected_size, int)
            or not isinstance(expected_digest, str)
        ):
            raise ValueError("OGDD array descriptor is incomplete.")
        if expected_size < 0 or expected_size > cls.MAX_MEMBER_BYTES:
            raise ValueError("OGDD array exceeds the supported size.")
        try:
            data = archive.read(member_name)
        except KeyError as error:
            raise ValueError(
                f"OGDD archive member is missing: {member_name}"
            ) from error
        if len(data) != expected_size or sha256(data).hexdigest() != expected_digest:
            raise ValueError(f"OGDD array failed integrity check: {member_name}")
        try:
            return np.load(BytesIO(data), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"OGDD array is invalid: {member_name}") from error

    @classmethod
    def _read_manifest(cls, archive: ZipFile) -> dict[str, Any]:
        try:
            manifest_data = archive.read("manifest.json")
        except KeyError as error:
            raise ValueError("OGDD archive has no manifest.json.") from error
        try:
            manifest = json.loads(manifest_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("OGDD manifest is not valid JSON.") from error
        if not isinstance(manifest, dict):
            raise ValueError("OGDD manifest must be a JSON object.")
        if manifest.get("format") != cls.FORMAT_NAME:
            raise ValueError("File does not contain an OGDD study.")
        if manifest.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("OGDD study version is not supported.")
        return manifest

    @classmethod
    def _validate_archive_members(cls, archive: ZipFile) -> None:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError("OGDD archive contains duplicate members.")
        for info in archive.infolist():
            if info.file_size > cls.MAX_MEMBER_BYTES:
                raise ValueError("OGDD archive member exceeds the size limit.")

    @classmethod
    def _validate_meshes(cls, meshes: dict[str, Mesh]) -> None:
        if not isinstance(meshes, dict):
            raise TypeError("OGDD meshes must be supplied as a mapping.")
        missing = cls.REQUIRED_MESHES.difference(meshes)
        if missing:
            raise ValueError(
                "OGDD study is missing required meshes: "
                + ", ".join(sorted(missing))
            )
        for name, mesh in meshes.items():
            if not isinstance(name, str) or not name:
                raise ValueError("OGDD mesh names must be non-empty strings.")
            if not isinstance(mesh, Mesh):
                raise TypeError(f"OGDD mesh {name} is not an OGDD Mesh.")
            if mesh.vertex_count == 0 or mesh.face_count == 0:
                raise ValueError(f"OGDD mesh {name} cannot be empty.")

    @staticmethod
    def _json_value(value: Any, label: str) -> Any:
        def default(item: Any) -> Any:
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, np.generic):
                return item.item()
            if isinstance(item, Path):
                return str(item)
            raise TypeError

        try:
            return json.loads(
                json.dumps(value, default=default, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise TypeError(f"{label} must contain JSON-compatible values.") from error
