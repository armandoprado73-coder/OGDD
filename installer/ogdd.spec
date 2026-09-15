"""PyInstaller definition for the standalone OGDD desktop application."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from pyvista import _vtk


PROJECT_ROOT = Path(SPECPATH).resolve().parent

def is_runtime_pyvista_module(name):
    """Exclude documentation, tests and optional web front ends."""

    excluded = (
        "pyvista.conftest",
        "pyvista.examples",
        "pyvista.ext",
        "pyvista.trame",
    )
    return not name.startswith(excluded)


vtk_module_names = (
    _vtk._CORE_MODULES | _vtk._PLOTTING_MODULES | _vtk._OPENGL_MODULES
)

datas = collect_data_files("pyvista") + collect_data_files("pyvistaqt") + [
    (str(PROJECT_ROOT / "LICENSE"), "."),
]
hiddenimports = (
    collect_submodules("pyvista", filter=is_runtime_pyvista_module)
    + collect_submodules("pyvistaqt")
    + [f"vtkmodules.{name}" for name in vtk_module_names]
    + ["vtkmodules.qt.QVTKRenderWindowInteractor"]
)

analysis = Analysis(
    [str(PROJECT_ROOT / "src" / "ogdd" / "app" / "__main__.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "setuptools"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="OGDD",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OGDD",
)
