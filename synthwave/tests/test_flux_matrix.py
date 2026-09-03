"""filament_flux_matrix must reproduce the per-filament Biot-Savart loop it replaced, and
modes sharing a rational surface must get their direct response from the shared flux
matrix and their own filament currents."""

import numpy as np
import pytest
import xarray as xr
from scipy.constants import mu_0

from synthwave.magnetic_geometry.filaments import (
    ToroidalFilamentTracer,
    filament_currents,
    filament_offsets,
)
from synthwave.mirnov.synthetic_signal import (
    direct_response_biot_savart,
    filament_flux_matrix,
)


def _sensors(rng, num_sensors):
    """Random sensor positions and unit normals a few meters out, assorted coil radii."""
    positions = rng.uniform(-3.0, 3.0, (num_sensors, 3))
    normals = rng.normal(size=(num_sensors, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return xr.Dataset(
        data_vars={
            "position": (("sensor_idx", "coord"), positions),
            "normal": (("sensor_idx", "coord"), normals),
            "radius": ("sensor_idx", rng.uniform(0.002, 0.02, num_sensors)),
        },
        coords={"sensor_idx": np.arange(num_sensors), "coord": ["x", "y", "z"]},
    )


def _random_filaments(rng):
    """Wobbly closed curves of assorted lengths, one with NaN points inside and one with
    a single valid point that must contribute zero flux."""
    filaments = []
    for num_points in (5, 50, 200):
        t = np.linspace(0, 2 * np.pi, num_points)
        curve = np.column_stack([np.cos(t), np.sin(2 * t), 0.3 * np.cos(3 * t)])
        filaments.append(curve + rng.normal(scale=0.05, size=curve.shape))
    with_nan = filaments[1].copy()
    with_nan[[3, 10, 11, 40], :] = np.nan
    filaments.append(with_nan)
    filaments.append(np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]]))
    return filaments


def _loop_flux_matrix(sensor_details, filament_list):
    """Unit-current flux per filament the way the per-filament loop computed it before
    filament_flux_matrix: one (n_sensors, n_seg, 3) Biot-Savart sum per filament."""
    sensor_positions = sensor_details["position"].data
    sensor_normals = sensor_details["normal"].data
    sensor_areas = np.pi * sensor_details["radius"].data ** 2
    flux = np.zeros((len(filament_list), len(sensor_positions)))
    for i, filament_pts in enumerate(filament_list):
        filament_pts = np.asarray(filament_pts, dtype=float)
        filament_pts = filament_pts[~np.isnan(filament_pts).any(axis=1)]
        if len(filament_pts) < 2:
            continue
        dl = filament_pts[1:] - filament_pts[:-1]
        midpoints = 0.5 * (filament_pts[1:] + filament_pts[:-1])
        r_prime = sensor_positions[:, None, :] - midpoints[None, :, :]
        r_prime_norm = np.linalg.norm(r_prime, axis=2)
        dl_cross_r = np.cross(dl[None, :, :], r_prime)
        B_total = np.sum(
            (mu_0 / (4 * np.pi)) * dl_cross_r / r_prime_norm[:, :, None] ** 3, axis=1
        )
        flux[i] = np.sum(B_total * sensor_normals, axis=1) * sensor_areas
    return flux


class TestFilamentFluxMatrix:
    def test_matches_per_filament_loop(self):
        rng = np.random.default_rng(0)
        sensors = _sensors(rng, 7)
        filaments = _random_filaments(rng)
        expected = _loop_flux_matrix(sensors, filaments)
        actual = filament_flux_matrix(sensors, filaments)
        assert actual.shape == (len(filaments), 7)
        assert np.all(actual[-1] == 0.0)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-11, atol=1e-13 * np.abs(expected).max()
        )

    @pytest.mark.parametrize("max_block_elements", [1, 7, 100, 10**9])
    def test_block_size_does_not_change_result(self, max_block_elements):
        rng = np.random.default_rng(1)
        sensors = _sensors(rng, 5)
        filaments = _random_filaments(rng)
        expected = _loop_flux_matrix(sensors, filaments)
        actual = filament_flux_matrix(
            sensors, filaments, max_block_elements=max_block_elements
        )
        np.testing.assert_allclose(
            actual, expected, rtol=1e-11, atol=1e-13 * np.abs(expected).max()
        )

    def test_direct_response_is_currents_dot_flux(self):
        rng = np.random.default_rng(2)
        sensors = _sensors(rng, 6)
        filaments = _random_filaments(rng)
        currents = rng.normal(size=len(filaments)) + 1j * rng.normal(
            size=len(filaments)
        )
        expected = currents @ filament_flux_matrix(sensors, filaments)
        actual = direct_response_biot_savart(sensors, filaments, list(currents))
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


class TestSharedSurface:
    NUM_FILAMENTS = 12
    BASE_NUM_POINTS = 60

    def _tracer(self, mode):
        return ToroidalFilamentTracer(
            mode=mode, R0=1.7, Z0=0.0, a=0.5, base_num_points=self.BASE_NUM_POINTS
        )

    @pytest.mark.parametrize(
        "base, harmonic",
        [
            ((1, 1), (2, 2)),
            ((2, 1), (4, 2)),
            ((2, 1), (6, 3)),
            ((3, 2), (6, 4)),
            ((2, -1), (4, -2)),
        ],
    )
    def test_harmonic_from_base_flux_matrix(self, base, harmonic):
        """A harmonic (k*m, k*n) traced on its own has the base filament geometry, and its
        direct response equals its own filament currents applied to the base flux matrix.
        """
        rng = np.random.default_rng(3)
        sensors = _sensors(rng, 9)
        base_filaments, _ = self._tracer(base).get_filament_list(self.NUM_FILAMENTS)
        harmonic_filaments, harmonic_currents = self._tracer(
            harmonic
        ).get_filament_list(self.NUM_FILAMENTS)
        np.testing.assert_array_equal(
            np.asarray(harmonic_filaments), np.asarray(base_filaments)
        )
        expected = direct_response_biot_savart(
            sensors, harmonic_filaments, harmonic_currents
        )
        actual = filament_currents(harmonic, self.NUM_FILAMENTS) @ filament_flux_matrix(
            sensors, base_filaments
        )
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-14 * np.abs(expected).max()
        )

    @pytest.mark.parametrize("mode", [(1, 1), (2, 1), (2, -1), (4, 2), (4, -2), (3, 2)])
    def test_filament_currents_match_dataset(self, mode):
        """filament_currents is the current vector get_filament_ds attaches to the filament
        copies, exp(i * n * phi_k), and filament_offsets their toroidal offsets."""
        ds = self._tracer(mode).get_filament_ds(
            self.NUM_FILAMENTS, coordinate_system="cylindrical"
        )
        currents = ds["current"].values
        np.testing.assert_array_equal(
            currents, filament_currents(mode, self.NUM_FILAMENTS)
        )
        np.testing.assert_allclose(
            ds["phi"].values[:, 0] - ds["phi"].values[0, 0],
            filament_offsets(self.NUM_FILAMENTS),
        )
        expected_step = np.exp(1j * 2 * np.pi * mode[1] / self.NUM_FILAMENTS)
        np.testing.assert_allclose(currents[1:] / currents[:-1], expected_step)
