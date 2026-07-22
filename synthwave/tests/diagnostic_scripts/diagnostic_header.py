"""Shared imports and path setup for diagnostic scripts.

These scripts are usually run directly from ``synthwave/tests/diagnostic_scripts``.
Running a file directly means relative imports like ``from .foo import bar`` do not
work, so this module adds the local SynthWave repository and the sibling TARS
repository to ``sys.path`` before importing project modules.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista
import xarray as xr

# Configure plotting
matplotlib.use("TkAgg")
plt.ion()
plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markeredgewidth"] = 2

try:
    from matplotlib import rc

    rc("font", **{"family": "serif", "serif": ["Palatino"]})
    rc("font", **{"size": 11})
    rc("text", usetex=True)
except Exception:
    pass

# Ensure TARS is in path
TARS_ROOT = "/mnt/home/rianc/Documents/TARS/"
if os.path.isdir(TARS_ROOT) and TARS_ROOT not in sys.path:
    sys.path.insert(0, TARS_ROOT)

# Ensure SynthWave is in path (diagnostic scripts are nested)
SYNTHWAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SYNTHWAVE_ROOT not in sys.path:
    sys.path.insert(0, SYNTHWAVE_ROOT)

# Common SynthWave imports
from synthwave.magnetic_geometry.equilibrium_field import convert_cocos, detect_cocos
from synthwave.magnetic_geometry.filaments import (
    EquilibriumField,
    EquilibriumFilamentTracer,
)

__all__ = [
    "np",
    "xr",
    "plt",
    "mtri",
    "pyvista",
    "convert_cocos",
    "detect_cocos",
    "EquilibriumField",
    "EquilibriumFilamentTracer",
    "os",
    "sys",
]
_THIS_FILE = Path(__file__).resolve()
_SYNTHWAVE_REPO = _THIS_FILE.parents[3]
_TARS_REPO = Path.home() / "Documents" / "TARS"

for _path in (_SYNTHWAVE_REPO, _TARS_REPO):
    if _path.exists():
        _path_str = str(_path)
        if _path_str not in sys.path:
            sys.path.insert(0, _path_str)


_tars_config = import_module("tars.config")
_tars_sensor_helpers = import_module("tars.out_of_scope.chisq_fast.sensor_helpers")
_tars_reconstruct_utils = import_module("tars.reconstruct.utils")

config = _tars_config.config
build_sensor_details_compat = _tars_sensor_helpers.build_sensor_details_compat
normalize_eq_field_dataset = _tars_sensor_helpers.normalize_eq_field_dataset
build_geqdsk = _tars_reconstruct_utils.build_geqdsk
direct_response_biot_savart = _tars_reconstruct_utils.direct_response_biot_savart
