"""The vectorized rational surface trace must reproduce the scalar marching Newton trace.

Both traces share the eta grid, so the comparison is pointwise.
"""

import os
from fractions import Fraction

import freeqdsk
import numpy as np
import pytest
import xarray as xr
from equilibrium_helpers import reference_trace_scalar_newton, tars_slice_to_eqdsk

from synthwave import PACKAGE_ROOT
from synthwave.magnetic_geometry.equilibrium_field import EquilibriumField
from synthwave.magnetic_geometry.filaments import (
    EquilibriumFilamentTracer,
    FilamentTraceError,
)

BASE_NUM_POINTS = 401
MODES = [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (3, 1), (2, -1)]
MAX_DISPLACEMENT = 1e-4  # [m]
MAX_DPHI = 1e-3  # [rad]

CMOD_EQDSK = os.path.join(PACKAGE_ROOT, "input_data", "cmod", "g1051202011.1000")
D3D_EQDSK = os.path.join(
    PACKAGE_ROOT,
    "..",
    "submodules",
    "OpenFUSIONToolkit",
    "examples",
    "TokaMaker",
    "DIIID",
    "g192185.02440",
)
TCV_INPUT = os.path.join(PACKAGE_ROOT, "tests", "test_data", "82878_tars_input.nc")
TCV_TIME_IDXS = (405, 805, 1205)


def load_eqdsk_field(path):
    with open(path) as f:
        return EquilibriumField(freeqdsk.geqdsk.read(f))


def load_tcv_field(time_idx):
    with xr.open_dataset(TCV_INPUT, engine="h5netcdf") as ds:
        ds_eq = ds.isel(time_idx=time_idx).load()
        cocos = int(ds.attrs["cocos"])
    return EquilibriumField(tars_slice_to_eqdsk(ds_eq), cocos_input=cocos)


EQUILIBRIA = {
    "cmod": (CMOD_EQDSK, lambda: load_eqdsk_field(CMOD_EQDSK)),
    "d3d": (D3D_EQDSK, lambda: load_eqdsk_field(D3D_EQDSK)),
    **{
        f"tcv_t{time_idx}": (
            TCV_INPUT,
            lambda time_idx=time_idx: load_tcv_field(time_idx),
        )
        for time_idx in TCV_TIME_IDXS
    },
}


@pytest.fixture(scope="module", params=list(EQUILIBRIA))
def equilibrium(request):
    path, loader = EQUILIBRIA[request.param]
    if not os.path.exists(path):
        pytest.skip(f"{request.param}: {path} is not available")
    return request.param, loader()


@pytest.mark.parametrize("mode", MODES)
def test_vectorized_trace_matches_scalar_reference(equilibrium, mode):
    name, eq_field = equilibrium
    tracer = EquilibriumFilamentTracer(mode, eq_field, base_num_points=BASE_NUM_POINTS)
    try:
        eq_field.get_psi_of_q(abs(Fraction(*mode)))
    except ValueError:
        pytest.skip(f"{name}: q={mode[0]}/{mode[1]} surface is absent")

    reference_points, reference_etas, _ = reference_trace_scalar_newton(
        eq_field, mode, tracer.num_points
    )
    points, etas = tracer.trace()

    np.testing.assert_array_equal(etas, reference_etas)
    assert points[0, 1] == 0.0, "phi must start at exactly zero"
    displacement = np.hypot(
        points[:, 0] - reference_points[:, 0], points[:, 2] - reference_points[:, 2]
    )
    assert displacement.max() < MAX_DISPLACEMENT, (
        f"{name} {mode}: max point displacement {displacement.max():.2e} m"
    )
    dphi = np.abs(points[:, 1] - reference_points[:, 1])
    assert dphi.max() < MAX_DPHI, (
        f"{name} {mode}: max phi difference {dphi.max():.2e} rad"
    )


@pytest.mark.skipif(
    not os.path.exists(TCV_INPUT),
    reason="Test requires TCV data which is not open source",
)
@pytest.mark.parametrize("time_idx, mode", [(1525, (3, 1)), (1445, (4, 3))])
def test_wrong_surface_is_a_failed_trace(time_idx, mode):
    """Late TCV slices where the q profile and the psi map disagree by 30 percent or more.

    The scalar reference rescaled these silently (integrated phi off by a factor 1.3 to 1.5).
    With the default closure_rtol they raise.
    """
    eq_field = load_tcv_field(time_idx)
    tracer = EquilibriumFilamentTracer(mode, eq_field, base_num_points=BASE_NUM_POINTS)
    _, _, phi_end_raw = reference_trace_scalar_newton(eq_field, mode, tracer.num_points)
    assert abs(abs(phi_end_raw) / (2 * np.pi) - abs(Fraction(*mode))) > 0.1 * abs(
        Fraction(*mode)
    ), "case no longer exercises the closure check, pick another slice"

    with pytest.raises(FilamentTraceError, match="misses"):
        tracer.trace()
    assert tracer.trace_cache == {}
    with pytest.raises(FilamentTraceError):
        tracer.get_filament_list(num_filaments=4)
