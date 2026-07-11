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

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista
import xarray as xr

_THIS_FILE = Path(__file__).resolve()
_SYNTHWAVE_REPO = _THIS_FILE.parents[3]
_TARS_REPO = Path.home() / "Documents" / "TARS"

for _path in (_SYNTHWAVE_REPO, _TARS_REPO):
    if _path.exists():
        _path_str = str(_path)
        if _path_str not in sys.path:
            sys.path.insert(0, _path_str)

from synthwave.magnetic_geometry.equilibrium_field import EquilibriumField, convert_cocos
from synthwave.magnetic_geometry.filaments import EquilibriumFilamentTracer

_tars_config = import_module("tars.config")
_tars_sensor_helpers = import_module("tars.out_of_scope.chisq_fast.sensor_helpers")
_tars_reconstruct_utils = import_module("tars.reconstruct.utils")

config = _tars_config.config
build_sensor_details_compat = _tars_sensor_helpers.build_sensor_details_compat
normalize_eq_field_dataset = _tars_sensor_helpers.normalize_eq_field_dataset
build_geqdsk = _tars_reconstruct_utils.build_geqdsk
direct_response_biot_savart = _tars_reconstruct_utils.direct_response_biot_savart