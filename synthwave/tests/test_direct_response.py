"""Several ways of calculating the direct response of filaments to sensors, should yield the same result

1: Using ThinCurr (compute_Msensor for coil->sensor Msc, no vessel solve)
2: Non-vectorized Biot-Savart (loop over filaments and sensors)
3: Vectorized Biot-Savart (direct_response_biot_savart, loop over filaments only)

Methods 2 and 3 must be numerically identical.
Method 1 should agree in phase structure with methods 2/3 (small sensor approximation).

Driven by the open C-Mod equilibrium and Mirnov sensor set under input_data/cmod.
"""

import os
import tempfile

import freeqdsk
import numpy as np
import pytest
import xarray as xr
from OpenFUSIONToolkit import OFT_env

from synthwave import PACKAGE_ROOT
from synthwave.magnetic_geometry.equilibrium_field import (
    EquilibriumField,
    biot_savart_cartesian,
)
from synthwave.magnetic_geometry.filaments import EquilibriumFilamentTracer
from synthwave.magnetic_geometry.utils import create_torus_mesh, wrapped_diff
from synthwave.mirnov.prep_thincurr_input import (
    gen_OFT_filament_and_eta_file,
    gen_OFT_sensors_file,
)
from synthwave.mirnov.synthetic_signal import (
    direct_response_biot_savart,
    direct_response_thincurr,
)

# ThinCurr / OFT_env hold global C++ state that cannot be shared across workers
pytestmark = pytest.mark.serial

_CMOD_DIR = os.path.join(PACKAGE_ROOT, "input_data", "cmod")
_CMOD_EQDSK_FILE = os.path.join(_CMOD_DIR, "g1051202011.1000")
_CMOD_SENSORS_FILE = os.path.join(_CMOD_DIR, "sensor_details_C_MOD_ALL.nc")

# time_idx=1100 is in the known 2/1 NTM window for the matching shot
_MODE = (2, 1)
_NUM_FILAMENTS = 23  # prime, coprime with n_local
_BASE_NUM_POINTS = 100


@pytest.fixture(scope="module")
def oft_env():
    return OFT_env(nthreads=2)


@pytest.fixture(scope="module")
def sensor_details():
    """Load the C-Mod Mirnov set and conform it to the OFT/Biot-Savart schema.

    The shipped file uses dim ``sensor`` and attr ``probe_set_name``; downstream
    helpers expect dim ``sensor_idx`` and attr ``sensor_set_name``.
    """
    ds = xr.open_dataset(_CMOD_SENSORS_FILE, engine="h5netcdf")
    ds = ds.rename({"sensor": "sensor_idx"})
    ds.attrs["sensor_set_name"] = ds.attrs["probe_set_name"]
    return ds


@pytest.fixture(scope="module")
def direct_response_inputs(sensor_details):
    """Build eq_field and trace the C-Mod filaments once for all methods."""
    with open(_CMOD_EQDSK_FILE, "r") as f:
        eqdsk = freeqdsk.geqdsk.read(f)
    eq_field = EquilibriumField(eqdsk)

    tracer = EquilibriumFilamentTracer(
        _MODE,
        eq_field=eq_field,
        base_num_points=_BASE_NUM_POINTS,
        default_trace_type=EquilibriumFilamentTracer.TraceType.SINGLE,
    )
    filament_list, current_list = tracer.get_filament_list(
        num_filaments=_NUM_FILAMENTS,
        coordinate_system="cartesian",
    )

    return {
        "eq_field": eq_field,
        "tracer": tracer,
        "sensor_details": sensor_details,
        "filament_list": filament_list,
        "current_list": current_list,
        "mode": _MODE,
    }


def direct_response_biot_savart_loop(
    sensor_details: xr.Dataset,
    filament_list: list,
    current_list: list,
) -> np.ndarray:
    """Non-vectorized Biot-Savart: loop over both filaments and sensors.

    Reference implementation used to verify the vectorized version.
    Returns complex array of shape (n_sensors,).
    """
    sensor_positions = sensor_details["position"].data
    sensor_normals = sensor_details["normal"].data
    sensor_areas = np.pi * sensor_details["radius"].data ** 2
    n_sensors = len(sensor_positions)

    direct_response = np.zeros(n_sensors, dtype=complex)
    for filament_pts, current in zip(filament_list, current_list):
        filament_pts = np.asarray(filament_pts, dtype=float)
        valid = ~np.isnan(filament_pts).any(axis=1)
        filament_pts = filament_pts[valid]
        if len(filament_pts) < 2:
            continue
        for k in range(n_sensors):
            B = biot_savart_cartesian(sensor_positions[k], filament_pts, 1.0)
            flux = np.dot(B, sensor_normals[k]) * sensor_areas[k]
            direct_response[k] += current * flux
    return direct_response


def test_vectorized_matches_loop(direct_response_inputs):
    """Vectorized and loop Biot-Savart must produce numerically identical results."""
    sensor_details = direct_response_inputs["sensor_details"]
    filament_list = direct_response_inputs["filament_list"]
    current_list = direct_response_inputs["current_list"]

    result_loop = direct_response_biot_savart_loop(
        sensor_details, filament_list, current_list
    )
    result_vec = direct_response_biot_savart(
        sensor_details, filament_list, current_list
    )

    np.testing.assert_allclose(
        result_loop,
        result_vec,
        rtol=1e-12,
        atol=0,
        err_msg="Vectorized Biot-Savart differs from loop version",
    )


def test_thincurr_phase_matches_biot_savart(oft_env, direct_response_inputs):
    """ThinCurr and Biot-Savart must agree on normalized phasor structure.

    Amplitude may differ (ThinCurr uses the Neumann formula over the sensor loop
    area; Biot-Savart approximates the sensor as a point times area). Phase should
    agree to within ~1% for the small C-Mod sensors (4 mm radius).
    """
    tracer = direct_response_inputs["tracer"]
    sensor_details = direct_response_inputs["sensor_details"]
    filament_list = direct_response_inputs["filament_list"]
    current_list = direct_response_inputs["current_list"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Vessel mesh is only needed so ThinCurr can set up; the direct
        # (coil->sensor) response does not depend on vessel currents.
        mesh_file = os.path.join(tmpdir, "mesh.h5")
        torus_mesh = create_torus_mesh(0.68, 0.3, ntheta=24, nphi=12)
        torus_mesh.write_to_file(mesh_file)

        # ThinCurr reads the filaments back from the XML, so it must be written
        # with the same filament count requested from the tracer.
        gen_OFT_filament_and_eta_file(
            working_directory=tmpdir,
            filament_list=filament_list,
            resistivity_list=[1e-6] * len(filament_list),
        )
        sensor_file_path = gen_OFT_sensors_file(
            sensor_details=sensor_details,
            working_directory=tmpdir,
        )

        ds_thincurr = direct_response_thincurr(
            oft_env=oft_env,
            tracer=tracer,
            mesh_file=mesh_file,
            sensor_details=sensor_details,
            sensor_file_path=sensor_file_path,
            working_directory=tmpdir,
        )

    tc = (
        ds_thincurr["direct_response_real"].data
        + 1j * ds_thincurr["direct_response_imag"].data
    )
    bs = direct_response_biot_savart(sensor_details, filament_list, current_list)

    # Normalize to the sensor with the most in-plane (smallest |Z|) normal
    ref_idx = int(np.argmin(np.abs(sensor_details["normal"].sel(coord="z").data)))

    def _normalize(z):
        z = z / (np.abs(z) + 1e-30)
        return z / (z[ref_idx] + 1e-30)

    tc_norm = _normalize(tc)
    bs_norm = _normalize(bs)

    phase_diff = np.abs(wrapped_diff(np.angle(tc_norm), np.angle(bs_norm)))
    assert phase_diff.mean() < 0.01, (
        f"Mean normalized phase difference ThinCurr vs Biot-Savart = "
        f"{phase_diff.mean():.4f}, expected < 0.01"
    )
