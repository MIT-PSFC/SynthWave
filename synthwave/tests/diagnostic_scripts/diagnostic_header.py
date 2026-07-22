import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Standard diagnostic stack
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

# Optional: LaTeX fonts (will fail if not installed, but usually fine in research envs)
# try:
#     from matplotlib import rc
#     rc("font", **{"family": "serif", "serif": ["Palatino"]})
#     rc("font", **{"size": 11})
#     rc("text", usetex=True)
# except Exception:
#     pass

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
