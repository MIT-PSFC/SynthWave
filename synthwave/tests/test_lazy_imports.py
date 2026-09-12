"""Importing the tracing and direct response modules must not load OpenFUSIONToolkit.

OpenFUSIONToolkit dlopens tens of MB of shared libraries at import time and
the plotting stack (vtk, pyvista) is similarly heavy, so both are imported inside the
ThinCurr functions only. Each module is imported in a fresh interpreter without
LD_LIBRARY_PATH, which is what a caller without the OFT environment sees.
"""

import os
import subprocess
import sys

import pytest

HEAVY_MODULES = ("OpenFUSIONToolkit", "vtk", "pyvista")

NO_VESSEL_MODULES = (
    "synthwave.magnetic_geometry.utils",
    "synthwave.magnetic_geometry.equilibrium_field",
    "synthwave.magnetic_geometry.filaments",
    "synthwave.mirnov.synthetic_signal",
    "synthwave.mirnov.prep_thincurr_input",
)


@pytest.mark.parametrize("module_name", NO_VESSEL_MODULES)
def test_import_does_not_load_heavy_modules(module_name):
    script = (
        f"import sys\n"
        f"import {module_name}\n"
        f"loaded = [name for name in {HEAVY_MODULES!r} if name in sys.modules]\n"
        "assert not loaded, f'heavy modules loaded: {loaded}'\n"
    )
    env = {key: val for key, val in os.environ.items() if key != "LD_LIBRARY_PATH"}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
