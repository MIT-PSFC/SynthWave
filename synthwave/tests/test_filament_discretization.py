"""Discretization order of the filament trace and the Biot-Savart direct response.

Two first-order errors used to dominate the direct response of high (m, n) modes:

1. direct_response_biot_savart used np.gradient for dl, which gives each end point
   a full segment on top of the half segments of the interior central differences.
   Every filament carried one extra segment of current at its ends. The ends sit at
   the outboard midplane next to the sensors and carry exp(i n phi0), so the extra
   elements form a spurious n-th harmonic ring current of order 1/num_points. For a
   4/3 mode at 401 base points it was 2.8x the true signal.
2. EquilibriumFilamentTracer.trace set phi_k = cumsum(d_phi) - d_phi[0], the sum of
   segments 1..k instead of 0..k-1, a poloidally varying error of order
   2 pi (m/n) / num_points.

With the segment-midpoint rule and the trapezoid phi integral both converge at second order.
The thresholds below have at least a 10x margin on the new implementation and
fail the old one by at least 10x (old values quoted in each docstring).
"""

import os

import freeqdsk
import numpy as np
import pytest
import xarray as xr
from scipy.constants import mu_0

from synthwave import PACKAGE_ROOT
from synthwave.magnetic_geometry.equilibrium_field import EquilibriumField
from synthwave.magnetic_geometry.filaments import EquilibriumFilamentTracer
from synthwave.mirnov.synthetic_signal import direct_response_biot_savart

_CMOD_DIR = os.path.join(PACKAGE_ROOT, "input_data", "cmod")
_CMOD_EQDSK_FILE = os.path.join(_CMOD_DIR, "g1051202011.1000")
_CMOD_SENSORS_FILE = os.path.join(_CMOD_DIR, "sensor_details_C_MOD_ALL.nc")

# Divisible by every n used below so filament ends meet the next filament's start
_NUM_FILAMENTS = 60


@pytest.fixture(scope="module")
def eq_field():
    with open(_CMOD_EQDSK_FILE, "r") as f:
        return EquilibriumField(freeqdsk.geqdsk.read(f))


@pytest.fixture(scope="module")
def sensor_details():
    ds = xr.open_dataset(_CMOD_SENSORS_FILE, engine="h5netcdf")
    return ds.rename({"sensor": "sensor_idx"})


def _point_sensors(positions, normals):
    """Minimal sensor_details for direct_response_biot_savart, unit area sensors."""
    positions = np.atleast_2d(positions)
    return xr.Dataset(
        data_vars={
            "position": (("sensor_idx", "coord"), positions),
            "normal": (("sensor_idx", "coord"), np.atleast_2d(normals)),
            "radius": ("sensor_idx", np.full(len(positions), 1.0 / np.sqrt(np.pi))),
        },
        coords={"sensor_idx": np.arange(len(positions)), "coord": ["x", "y", "z"]},
    )


def _circular_loop(radius, num_points):
    """Closed loop in the z = 0 plane, last point repeats the first."""
    theta = np.linspace(0, 2 * np.pi, num_points + 1)
    return np.column_stack(
        (radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta))
    )


def _pattern_error(response, reference):
    """Relative residual after removing the global complex scale [fraction].

    Only the sensor-to-sensor cross phase pattern matters for spectral analysis.
    """
    scale = np.vdot(reference, response) / np.vdot(reference, reference)
    return np.linalg.norm(response - scale * reference) / np.linalg.norm(response)


def test_biot_savart_open_arcs_equal_closed_loop():
    """Arcs sharing end points must sum to the loop they make up, to round-off.

    The mode filaments are open arcs (one poloidal turn) whose ends coincide with
    the next filament's start, so the sum over filaments must not depend on where
    the polyline is cut.

    np.gradient counted one extra segment per cut, a 4 percent difference for 12 arcs of 20 segments.
    """
    num_segments, num_arcs = 240, 12
    loop = _circular_loop(0.5, num_segments)
    per_arc = num_segments // num_arcs
    arcs = [loop[k * per_arc : (k + 1) * per_arc + 1] for k in range(num_arcs)]
    sensors = _point_sensors(
        [[0.2, 0.1, 0.3], [0.7, 0.2, 0.15]],
        [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]],
    )

    whole = direct_response_biot_savart(sensors, [loop], [1.0])
    split = direct_response_biot_savart(sensors, arcs, [1.0] * num_arcs)

    np.testing.assert_allclose(split, whole, rtol=1e-12, atol=0)


def test_biot_savart_circular_loop_second_order():
    """On-axis field of a circular loop, B = mu_0 I a^2 / (2 (a^2 + z^2)^(3/2)).

    Segment-midpoint rule: relative error 1.3e-4 at 256 segments, reduced x4 per doubling of elements.
    np.gradient: relative error 3.8e-3 at 256 segments, reduced x2 per doubling of elements.
    """
    radius, z = 0.5, 0.0
    sensor = _point_sensors([[0.0, 0.0, z]], [[0.0, 0.0, 1.0]])
    exact = mu_0 * radius**2 / (2 * (radius**2 + z**2) ** 1.5)

    errors = {}
    for num_segments in (128, 256):
        loop = _circular_loop(radius, num_segments)
        response = direct_response_biot_savart(sensor, [loop], [1.0])
        errors[num_segments] = abs(float(response[0].real) / exact - 1)

    assert errors[256] < 5e-4, f"relative error {errors[256]:.2e} at 256 segments"
    assert errors[128] / errors[256] > 3, (
        f"convergence ratio {errors[128] / errors[256]:.2f}, expected ~4 (second order)"
    )


@pytest.mark.parametrize("mode", [(2, 1), (3, 2)])
def test_trace_phi_second_order(eq_field, mode):
    """Toroidal angle along the traced filament converges at second order in points.

    Compared on a common eta grid against a 3200 base-point trace.
    Trapezoid phi: max error 0.009 deg at 200 base points, ratio 4 per doubling.
    The shifted cumsum gave 1.9 deg at 200 points (about 2 pi (m/n) / num_points), ratio 2.
    """
    eta_grid = np.linspace(0.01, 2 * np.pi - 0.01, 3000)

    def phi_on_grid(base_num_points):
        tracer = EquilibriumFilamentTracer(
            mode, eq_field=eq_field, base_num_points=base_num_points
        )
        points, etas = tracer.trace()
        assert points[0, 1] == 0.0, "phi must start at exactly zero"
        return np.interp(eta_grid, etas, points[:, 1])

    reference = phi_on_grid(3200)
    errors = {
        base: np.degrees(np.abs(phi_on_grid(base) - reference).max())
        for base in (200, 400)
    }

    assert errors[200] < 0.1, f"max phi error {errors[200]:.3f} deg at 200 base points"
    assert errors[200] / errors[400] > 3, (
        f"convergence ratio {errors[200] / errors[400]:.2f}, expected ~4 (second order)"
    )


def test_direct_response_second_order_in_points(eq_field, sensor_details):
    """End-to-end: the sensor pattern of a 5/4 mode converges at second order.

    High (m, n) modes have a small true signal at the wall, so the spurious end-point ring current dominated them.
    
    Against a 3200 base-point reference with 60 filaments, 
    the pattern error at 200 base points is 0.06 percent now (was 17.5 percent).
    """
    mode = (5, 4)

    def response(base_num_points):
        tracer = EquilibriumFilamentTracer(
            mode, eq_field=eq_field, base_num_points=base_num_points
        )
        filament_list, current_list = tracer.get_filament_list(
            num_filaments=_NUM_FILAMENTS, coordinate_system="cartesian"
        )
        return direct_response_biot_savart(sensor_details, filament_list, current_list)

    reference = response(3200)
    errors = {base: _pattern_error(response(base), reference) for base in (200, 400)}

    assert errors[200] < 0.01, f"pattern error {100 * errors[200]:.3f} % at 200 base points"
    assert errors[200] / errors[400] > 3, (
        f"convergence ratio {errors[200] / errors[400]:.2f}, expected ~4 (second order)"
    )
