import os
from fractions import Fraction

import freeqdsk
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sympy import nextprime

from synthwave import PACKAGE_ROOT
from synthwave.magnetic_geometry.equilibrium_field import EquilibriumField
from synthwave.magnetic_geometry.filaments import (
    EquilibriumFilamentTracer,
    FilamentTracer,
    ToroidalFilamentTracer,
)

FIG_DIR = os.path.join(PACKAGE_ROOT, "tests", "figures")
_CMOD_EQDSK_FILE = os.path.join(PACKAGE_ROOT, "input_data", "cmod", "g1051202011.1000")


@pytest.fixture(scope="module")
def cmod_eqdsk():
    with open(_CMOD_EQDSK_FILE, "r") as f:
        return freeqdsk.geqdsk.read(f)


class TestToroidalFilamentTracer:
    """Test the ToroidalFilamentTracer class."""

    fig_dir = os.path.join(FIG_DIR, "test_toroidal_filament_tracer")

    @pytest.mark.parametrize("scale_points", [True, False])
    @pytest.mark.parametrize("prevent_synthetic_structure", [True, False])
    def test_init(self, scale_points, prevent_synthetic_structure):
        """Test initialization of ToroidalFilamentTracer."""
        m, n = 2, 1
        R0, Z0, a = 1.8, 0.0, 0.5
        base_num_points = 100

        tracer = ToroidalFilamentTracer(
            m,
            n,
            R0,
            Z0,
            a,
            base_num_points,
            scale_points=scale_points,
            prevent_synthetic_structure=prevent_synthetic_structure,
        )

        assert tracer.m == m
        assert tracer.n == n
        assert tracer.R0 == R0
        assert tracer.Z0 == Z0
        assert tracer.a == a

        expected_points = base_num_points
        if scale_points:
            expected_points = int(base_num_points * m / n)
        if prevent_synthetic_structure:
            expected_points = nextprime(expected_points)

        assert tracer.num_points == expected_points

    @pytest.mark.parametrize("scale_points", [True, False])
    @pytest.mark.parametrize("prevent_synthetic_structure", [True, False])
    def test_trace_basic(self, scale_points, prevent_synthetic_structure):
        """Test basic tracing functionality."""
        m, n = 2, 1
        R0, Z0, a = 1.8, 0.0, 0.5
        base_num_points = 100

        filament = ToroidalFilamentTracer(
            m,
            n,
            R0,
            Z0,
            a,
            base_num_points,
            scale_points=scale_points,
            prevent_synthetic_structure=prevent_synthetic_structure,
        )
        points, etas = filament.trace()

        # Check shape
        expected_points = base_num_points
        if scale_points:
            expected_points = int(base_num_points * m / n)
        if prevent_synthetic_structure:
            expected_points = nextprime(expected_points)

        assert points.shape[1] == 3
        assert points.shape[0] == expected_points
        assert etas.shape[0] == points.shape[0]

    @pytest.mark.parametrize("R0", [1.5, 2.0, 2.5])
    @pytest.mark.parametrize("Z0", [0.0, 0.1, -0.1])
    @pytest.mark.parametrize("a", [0.4, 0.5, 0.6])
    @pytest.mark.parametrize("mode", [(1, 1), (2, 1), (3, 2), (3, 1)])
    def test_trace_circular_geometry(self, R0, Z0, a, mode):
        """Test that the filament traces a circle in the poloidal plane."""
        m, n = mode
        base_num_points = 100

        tracer = ToroidalFilamentTracer(
            m,
            n,
            R0,
            Z0,
            a,
            base_num_points,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        points, etas = tracer.trace()

        R, phi, Z = points[:, 0], points[:, 1], points[:, 2]

        # Check that points form a circle in R-Z plane
        # Distance from magnetic axis should be constant (= minor radius)
        distances = np.sqrt((R - R0) ** 2 + (Z - Z0) ** 2)
        np.testing.assert_allclose(distances, a, rtol=1e-10)

        # Check that we complete the correct number of toroidal turns
        phi_range = phi[-1] - phi[0]
        expected_phi_range = 2 * np.pi * m / n
        np.testing.assert_allclose(phi_range, expected_phi_range, rtol=1e-10)

        np.testing.assert_allclose(
            etas, np.linspace(0, 2 * np.pi, base_num_points), rtol=1e-10
        )

    def test_trace_specific_points_2_1(self):
        """Test that specific points are at expected locations."""
        m, n = 2, 1
        R0, Z0, a = 1.8, 0.0, 0.5
        num_points = 9  # Use 9 points for easy checking. This gives points at 0, π/4, π/2, ..., 2π

        tracer = ToroidalFilamentTracer(
            m,
            n,
            R0,
            Z0,
            a,
            num_points,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        points, etas = tracer.trace()

        R, phi, Z = points[:, 0], points[:, 1], points[:, 2]

        # First point should be at outboard midplane
        np.testing.assert_allclose(R[0], R0 + a, atol=1e-10)
        np.testing.assert_allclose(Z[0], Z0, atol=1e-10)
        np.testing.assert_allclose(phi[0], 0.0, atol=1e-10)

        # Check a few specific points
        # At 1/4 of the way around poloidal angle (π/2), this should be phi = π
        quarter_idx = 2
        expected_R_quarter = R0  # At top of circle
        expected_Z_quarter = Z0 + a
        np.testing.assert_allclose(R[quarter_idx], expected_R_quarter, atol=1e-10)
        np.testing.assert_allclose(Z[quarter_idx], expected_Z_quarter, atol=1e-10)
        np.testing.assert_allclose(phi[quarter_idx], np.pi, atol=1e-10)

        # At 1/2 of the way around poloidal angle (π), this should be phi = 2π
        half_idx = 4
        expected_R_half = R0 - a  # At inboard midplane
        expected_Z_half = Z0
        np.testing.assert_allclose(R[half_idx], expected_R_half, atol=1e-10)
        np.testing.assert_allclose(Z[half_idx], expected_Z_half, atol=1e-10)
        np.testing.assert_allclose(phi[half_idx], 2 * np.pi, atol=1e-10)

    def test_trace_specific_points_3_2(self):
        """Test that specific points are at expected locations for m=3, n=2."""
        m, n = 3, 2
        R0, Z0, a = 1.8, 0.0, 0.5
        num_points = 13  # Use 13 points for easy checking. This gives points at 0, π/6, π/3, ..., 2π

        tracer = ToroidalFilamentTracer(
            m,
            n,
            R0,
            Z0,
            a,
            num_points,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        points, etas = tracer.trace()

        R, phi, Z = points[:, 0], points[:, 1], points[:, 2]

        # First point should be at outboard midplane
        np.testing.assert_allclose(R[0], R0 + a, atol=1e-10)
        np.testing.assert_allclose(Z[0], Z0, atol=1e-10)
        np.testing.assert_allclose(phi[0], 0.0, atol=1e-10)

        # At 1/6 of the way around poloidal angle (π/3), this should be phi = π/2
        sixth_idx = 2
        expected_R_sixth = R0 + a * np.cos(np.pi / 3)
        expected_Z_sixth = Z0 + a * np.sin(np.pi / 3)
        np.testing.assert_allclose(R[sixth_idx], expected_R_sixth, atol=1e-10)
        np.testing.assert_allclose(Z[sixth_idx], expected_Z_sixth, atol=1e-10)
        np.testing.assert_allclose(phi[sixth_idx], np.pi / 2, atol=1e-10)

        # At 1/2 the way around poloidal angle (π), this should be phi = 3π/2
        half_idx = 6
        expected_R_half = R0 - a  # At inboard midplane
        expected_Z_half = Z0
        np.testing.assert_allclose(R[half_idx], expected_R_half, atol=1e-10)
        np.testing.assert_allclose(Z[half_idx], expected_Z_half, atol=1e-10)
        np.testing.assert_allclose(phi[half_idx], 3 * np.pi / 2, atol=1e-10)

    def test_trace_different_num_points(self):
        """Test tracing with different number of points."""
        m, n = 1, 1
        R0, Z0, a = 2.0, 0.0, 0.5

        tracer = ToroidalFilamentTracer(
            m,
            n,
            R0,
            Z0,
            a,
            base_num_points=100,
            scale_points=False,
            prevent_synthetic_structure=False,
        )

        # Test with default number of points
        points_default, _ = tracer.trace()
        assert points_default.shape[0] == 100

        # Test with custom number of points
        points_custom, _ = tracer.trace(num_points=200)
        assert points_custom.shape[0] == 200

        # Both should have same geometry, just different resolution
        R_default, _phi_default, Z_default = (
            points_default[:, 0],
            points_default[:, 1],
            points_default[:, 2],
        )
        R_custom, _phi_custom, Z_custom = (
            points_custom[:, 0],
            points_custom[:, 1],
            points_custom[:, 2],
        )

        # Check that distances from axis are still correct
        distances_default = np.sqrt((R_default - R0) ** 2 + (Z_default - Z0) ** 2)
        distances_custom = np.sqrt((R_custom - R0) ** 2 + (Z_custom - Z0) ** 2)

        np.testing.assert_allclose(distances_default, a, rtol=1e-10)
        np.testing.assert_allclose(distances_custom, a, rtol=1e-10)

    @pytest.mark.parametrize(
        "mode",
        [
            {"m": 1, "n": 1},
            {"m": 2, "n": 1},
            {"m": 3, "n": 2},
            {"m": 3, "n": 1},
            {"m": 4, "n": 3},
            {"m": 5, "n": 4},
            {"m": 4, "n": 1},
            {"m": 5, "n": 1},
            {"m": -3, "n": 2},
        ],
    )
    def test_points_and_currents(self, mode):
        """Test get_filament_ds method for correct output dataset."""
        major_radius = 1
        minor_radius = 0.3
        num_filaments = 19

        fig_dir = os.path.join(self.fig_dir, "test_points_and_currents")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        toroidal_tracer = ToroidalFilamentTracer(
            mode["m"], mode["n"], major_radius, 0, minor_radius, base_num_points=100
        )

        filament_ds = toroidal_tracer.get_filament_ds(
            num_filaments=num_filaments, coordinate_system="toroidal"
        )

        # Create plot of filament points in phi and eta coordinates
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Extract data
        phi_vals = filament_ds["phi"].values  # Shape: (num_filaments, num_points)
        eta_vals = filament_ds["eta"].values  # Shape: (num_points,)
        currents = filament_ds[
            "current"
        ].values  # Shape: (num_filaments,), complex values

        # Determine the phi range for appropriate x-axis ticks
        phi_max = phi_vals.max()

        # Create x-axis ticks that span the entire phi domain
        # Number of ticks based on the range (use pi/2 increments)
        num_x_ticks = int(np.ceil(phi_max / (np.pi / 2))) + 1
        x_ticks = [i * np.pi / 2 for i in range(num_x_ticks)]
        x_labels = []
        for tick in x_ticks:
            # Convert to fraction of pi
            ratio = tick / np.pi
            frac = Fraction(int(ratio * 2), 2).limit_denominator()
            if frac.denominator == 1:
                x_labels.append(f"${frac.numerator}\pi$")
            else:
                x_labels.append(f"${frac.numerator}\pi/{frac.denominator}$")

        # Plot real component of current
        scatter1 = ax1.scatter(
            phi_vals.flatten(),
            np.tile(eta_vals, num_filaments),
            c=np.repeat(currents.real, phi_vals.shape[1]),
            cmap="RdBu_r",
            s=20,
            alpha=0.6,
        )
        ax1.set_xlabel("phi")
        ax1.set_ylabel("eta")
        ax1.set_title(f"Real Component of Current (m={mode['m']}, n={mode['n']})")
        ax1.grid(True, alpha=0.3)

        # Set x-axis ticks with pi fractions
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels)

        # Set y-axis ticks with pi fractions
        ax1.set_yticks(
            [
                0,
                np.pi / 6,
                np.pi / 4,
                np.pi / 3,
                np.pi / 2,
                2 * np.pi / 3,
                3 * np.pi / 4,
                5 * np.pi / 6,
                np.pi,
                7 * np.pi / 6,
                5 * np.pi / 4,
                4 * np.pi / 3,
                3 * np.pi / 2,
                5 * np.pi / 3,
                7 * np.pi / 4,
                11 * np.pi / 6,
                2 * np.pi,
            ]
        )
        ax1.set_yticklabels(
            [
                "0",
                r"$\pi/6$",
                r"$\pi/4$",
                r"$\pi/3$",
                r"$\pi/2$",
                r"$2\pi/3$",
                r"$3\pi/4$",
                r"$5\pi/6$",
                r"$\pi$",
                r"$7\pi/6$",
                r"$5\pi/4$",
                r"$4\pi/3$",
                r"$3\pi/2$",
                r"$5\pi/3$",
                r"$7\pi/4$",
                r"$11\pi/6$",
                r"$2\pi$",
            ]
        )

        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label("Re(I)")

        # Plot imaginary component of current
        scatter2 = ax2.scatter(
            phi_vals.flatten(),
            np.tile(eta_vals, num_filaments),
            c=np.repeat(currents.imag, phi_vals.shape[1]),
            cmap="RdBu_r",
            s=20,
            alpha=0.6,
        )
        ax2.set_xlabel("phi")
        ax2.set_ylabel("eta")
        ax2.set_title(f"Imaginary Component of Current (m={mode['m']}, n={mode['n']})")
        ax2.grid(True, alpha=0.3)

        # Set x-axis ticks with pi fractions (same as ax1)
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels)

        # Set y-axis ticks with pi fractions
        ax2.set_yticks(
            [
                0,
                np.pi / 6,
                np.pi / 4,
                np.pi / 3,
                np.pi / 2,
                2 * np.pi / 3,
                3 * np.pi / 4,
                5 * np.pi / 6,
                np.pi,
                7 * np.pi / 6,
                5 * np.pi / 4,
                4 * np.pi / 3,
                3 * np.pi / 2,
                5 * np.pi / 3,
                7 * np.pi / 4,
                11 * np.pi / 6,
                2 * np.pi,
            ]
        )
        ax2.set_yticklabels(
            [
                "0",
                r"$\pi/6$",
                r"$\pi/4$",
                r"$\pi/3$",
                r"$\pi/2$",
                r"$2\pi/3$",
                r"$3\pi/4$",
                r"$5\pi/6$",
                r"$\pi$",
                r"$7\pi/6$",
                r"$5\pi/4$",
                r"$4\pi/3$",
                r"$3\pi/2$",
                r"$5\pi/3$",
                r"$7\pi/4$",
                r"$11\pi/6$",
                r"$2\pi$",
            ]
        )

        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label("Im(I)")

        fig.tight_layout()
        fig_path = os.path.join(fig_dir, f"m{mode['m']}_n{mode['n']}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        # All current magnitudes must be 1 (unit phasors)
        np.testing.assert_allclose(np.abs(currents), 1.0, rtol=1e-10)

        # Adjacent filaments must have equal phase steps of 2*pi*sign(m)*n_local/num_filaments
        # Wrap expected to (-pi, pi] to match np.angle output range
        ratio = Fraction(mode["m"], mode["n"])
        n_local = ratio.denominator
        m_sign = int(np.sign(ratio.numerator)) if ratio.numerator != 0 else 1
        expected_phase_step = np.angle(
            np.exp(1j * 2 * np.pi * m_sign * n_local / num_filaments)
        )
        measured_phase_steps = np.angle(currents[1:] / currents[:-1])
        np.testing.assert_allclose(
            measured_phase_steps, expected_phase_step, atol=1e-10
        )

    @pytest.mark.parametrize(
        "mode",
        [
            {"m": 2, "n": 1},
            {"m": 3, "n": 2},
            {"m": -3, "n": 2},
            {"m": 3, "n": 1},
            {"m": 4, "n": 3},
        ],
    )
    def test_points_and_currents_3d(self, mode):
        major_radius = 1
        minor_radius = 0.3
        num_filaments = 7

        fig_dir = os.path.join(self.fig_dir, "test_points_and_currents_3d")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        toroidal_tracer = ToroidalFilamentTracer(
            mode["m"], mode["n"], major_radius, 0, minor_radius, base_num_points=100
        )

        filament_ds = toroidal_tracer.get_filament_ds(
            num_filaments=num_filaments, coordinate_system="cartesian"
        )

        # Create 3D plot of filament traces
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Extract data
        X = filament_ds["x"].values  # Shape: (num_filaments, num_points)
        Y = filament_ds["y"].values  # Shape: (num_filaments, num_points)
        Z = filament_ds["z"].values  # Shape: (num_filaments, num_points)
        currents = filament_ds["current"].values  # Shape: (num_filaments

        # Plot each filament with color based on real component of current
        norm = plt.Normalize(vmin=currents.real.min(), vmax=currents.real.max())
        cmap = plt.cm.viridis

        for i in range(num_filaments):
            color = cmap(norm(currents.real[i]))
            ax.plot(X[i, :], Y[i, :], Z[i, :], color=color, linewidth=4, alpha=0.7)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("|I| (Real component of current)", fontsize=12)

        # Set labels and title
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_zlabel("Z [m]", fontsize=12)
        ax.set_title(f"3D Filament Traces (m={mode['m']}, n={mode['n']})", fontsize=14)

        # Make the aspect ratio equal on all 3 axes
        max_range = (
            np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
            / 2.0
        )
        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set viewing angle
        ax.view_init(elev=20, azim=45)

        fig.tight_layout()
        fig_path = os.path.join(fig_dir, f"m{mode['m']}_n{mode['n']}_3d.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        # All current magnitudes must be 1 (unit phasors)
        np.testing.assert_allclose(np.abs(currents), 1.0, rtol=1e-10)

        # Dataset shape must match num_filaments x num_points
        assert filament_ds["x"].shape == (num_filaments, toroidal_tracer.num_points)
        assert filament_ds["y"].shape == (num_filaments, toroidal_tracer.num_points)
        assert filament_ds["z"].shape == (num_filaments, toroidal_tracer.num_points)


class TestEquilibriumFilamentTracer:
    """Test the EquilibriumFilamentTracer class."""

    fig_dir = os.path.join(FIG_DIR, "test_equilibrium_filament_tracer")

    @pytest.mark.parametrize("scale_points", [True, False])
    @pytest.mark.parametrize("prevent_synthetic_structure", [True, False])
    def test_init(self, cmod_eqdsk, scale_points, prevent_synthetic_structure):
        """Test initialization of EquilibriumFilamentTracer."""

        eq_field = EquilibriumField(cmod_eqdsk)

        m, n = 2, 1
        base_num_points = 100
        tracer = EquilibriumFilamentTracer(
            m,
            n,
            eq_field,
            base_num_points,
            scale_points=scale_points,
            prevent_synthetic_structure=prevent_synthetic_structure,
        )

        assert tracer.m == m
        assert tracer.n == n
        assert tracer.eq_field == eq_field

        expected_points = base_num_points
        if scale_points:
            expected_points = int(base_num_points * m / n)
        if prevent_synthetic_structure:
            expected_points = nextprime(expected_points)

        assert tracer.num_points == expected_points

    @pytest.mark.parametrize("scale_points", [True, False])
    @pytest.mark.parametrize("prevent_synthetic_structure", [True, False])
    def test_trace_basic(self, cmod_eqdsk, scale_points, prevent_synthetic_structure):
        """Test basic tracing functionality."""
        eq_field = EquilibriumField(cmod_eqdsk)

        m, n = 2, 1
        base_num_points = 100
        tracer = EquilibriumFilamentTracer(
            m,
            n,
            eq_field,
            base_num_points=base_num_points,
            scale_points=scale_points,
            prevent_synthetic_structure=prevent_synthetic_structure,
        )
        points, etas = tracer.trace()

        expected_points = base_num_points
        if scale_points:
            expected_points = int(base_num_points * m / n)
        if prevent_synthetic_structure:
            expected_points = nextprime(expected_points)

        # Check shape
        assert points.shape == (expected_points, 3)
        assert etas.shape == (expected_points,)

    @pytest.mark.parametrize(
        "mode",
        [
            {"m": 1, "n": 1},
            {"m": 2, "n": 1},
            {"m": 3, "n": 2},
            {"m": 3, "n": 1},
            {"m": 4, "n": 3},
            {"m": 5, "n": 4},
            {"m": 4, "n": 1},
            {"m": 5, "n": 1},
            {"m": -3, "n": 2},
        ],
    )
    def test_points_and_currents(self, cmod_eqdsk, mode):
        """Test get_filament_ds method for correct output dataset."""

        eq_field = EquilibriumField(cmod_eqdsk)
        num_filaments = 23

        fig_dir = os.path.join(self.fig_dir, "test_points_and_currents")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        equilibrium_tracer = EquilibriumFilamentTracer(
            mode["m"], mode["n"], eq_field, base_num_points=100
        )

        filament_ds = equilibrium_tracer.get_filament_ds(
            num_filaments=num_filaments, coordinate_system="toroidal"
        )

        # Create plot of filament points in phi and eta coordinates
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Extract data
        phi_vals = filament_ds["phi"].values  # Shape: (num_filaments, num_points)
        eta_vals = filament_ds["eta"].values  # Shape: (num_points,)
        currents = filament_ds[
            "current"
        ].values  # Shape: (num_filaments,), complex values

        # Determine the phi range for appropriate x-axis ticks
        phi_min = phi_vals.min()
        phi_max = phi_vals.max()

        # Create x-axis ticks that span the entire phi domain
        # Number of ticks based on the range (use pi/2 increments)
        # Calculate starting and ending tick indices to handle negative phi values
        start_tick_idx = int(np.floor(phi_min / (np.pi / 2)))
        end_tick_idx = int(np.ceil(phi_max / (np.pi / 2)))
        x_ticks = [i * np.pi / 2 for i in range(start_tick_idx, end_tick_idx + 1)]
        x_labels = []
        for tick in x_ticks:
            # Convert to fraction of pi
            ratio = tick / np.pi
            frac = Fraction(int(ratio * 2), 2).limit_denominator()
            if frac.denominator == 1:
                if frac.numerator > 0:
                    x_labels.append(f"${frac.numerator}\pi$")
                else:
                    x_labels.append(f"${frac.numerator}\pi$")
            else:
                if frac.numerator > 0:
                    x_labels.append(f"${frac.numerator}\pi/{frac.denominator}$")
                else:
                    x_labels.append(f"$-{abs(frac.numerator)}\pi/{frac.denominator}$")

        # Plot real component of current
        scatter1 = ax1.scatter(
            phi_vals.flatten(),
            np.tile(eta_vals, num_filaments),
            c=np.repeat(currents.real, phi_vals.shape[1]),
            cmap="RdBu_r",
            s=20,
            alpha=0.6,
        )
        ax1.set_xlabel("phi")
        ax1.set_ylabel("eta")
        ax1.set_title(f"Real Component of Current (m={mode['m']}, n={mode['n']})")
        ax1.grid(True, alpha=0.3)

        # Set x-axis ticks with pi fractions
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels)

        # Set y-axis ticks with pi fractions
        ax1.set_yticks(
            [
                0,
                np.pi / 6,
                np.pi / 4,
                np.pi / 3,
                np.pi / 2,
                2 * np.pi / 3,
                3 * np.pi / 4,
                5 * np.pi / 6,
                np.pi,
                7 * np.pi / 6,
                5 * np.pi / 4,
                4 * np.pi / 3,
                3 * np.pi / 2,
                5 * np.pi / 3,
                7 * np.pi / 4,
                11 * np.pi / 6,
                2 * np.pi,
            ]
        )
        ax1.set_yticklabels(
            [
                "0",
                r"$\pi/6$",
                r"$\pi/4$",
                r"$\pi/3$",
                r"$\pi/2$",
                r"$2\pi/3$",
                r"$3\pi/4$",
                r"$5\pi/6$",
                r"$\pi$",
                r"$7\pi/6$",
                r"$5\pi/4$",
                r"$4\pi/3$",
                r"$3\pi/2$",
                r"$5\pi/3$",
                r"$7\pi/4$",
                r"$11\pi/6$",
                r"$2\pi$",
            ]
        )

        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label("Re(I)")

        # Plot imaginary component of current
        scatter2 = ax2.scatter(
            phi_vals.flatten(),
            np.tile(eta_vals, num_filaments),
            c=np.repeat(currents.imag, phi_vals.shape[1]),
            cmap="RdBu_r",
            s=20,
            alpha=0.6,
        )
        ax2.set_xlabel("phi")
        ax2.set_ylabel("eta")
        ax2.set_title(f"Imaginary Component of Current (m={mode['m']}, n={mode['n']})")
        ax2.grid(True, alpha=0.3)

        # Set x-axis ticks with pi fractions (same as ax1)
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels)

        # Set y-axis ticks with pi fractions
        ax2.set_yticks(
            [
                0,
                np.pi / 6,
                np.pi / 4,
                np.pi / 3,
                np.pi / 2,
                2 * np.pi / 3,
                3 * np.pi / 4,
                5 * np.pi / 6,
                np.pi,
                7 * np.pi / 6,
                5 * np.pi / 4,
                4 * np.pi / 3,
                3 * np.pi / 2,
                5 * np.pi / 3,
                7 * np.pi / 4,
                11 * np.pi / 6,
                2 * np.pi,
            ]
        )
        ax2.set_yticklabels(
            [
                "0",
                r"$\pi/6$",
                r"$\pi/4$",
                r"$\pi/3$",
                r"$\pi/2$",
                r"$2\pi/3$",
                r"$3\pi/4$",
                r"$5\pi/6$",
                r"$\pi$",
                r"$7\pi/6$",
                r"$5\pi/4$",
                r"$4\pi/3$",
                r"$3\pi/2$",
                r"$5\pi/3$",
                r"$7\pi/4$",
                r"$11\pi/6$",
                r"$2\pi$",
            ]
        )

        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label("Im(I)")

        fig.tight_layout()
        fig_path = os.path.join(fig_dir, f"m{mode['m']}_n{mode['n']}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        # All current magnitudes must be 1 (unit phasors)
        np.testing.assert_allclose(np.abs(currents), 1.0, rtol=1e-10)

        # Adjacent filaments must have equal phase steps of 2*pi*sign(m)*n_local/num_filaments
        # Wrap expected to (-pi, pi] to match np.angle output range
        ratio = Fraction(mode["m"], mode["n"])
        n_local = ratio.denominator
        m_sign = int(np.sign(ratio.numerator)) if ratio.numerator != 0 else 1
        expected_phase_step = np.angle(
            np.exp(1j * 2 * np.pi * m_sign * n_local / num_filaments)
        )
        measured_phase_steps = np.angle(currents[1:] / currents[:-1])
        np.testing.assert_allclose(
            measured_phase_steps, expected_phase_step, atol=1e-10
        )

    @pytest.mark.parametrize(
        "mode", [{"m": 2, "n": 1}, {"m": 3, "n": 2}, {"m": 3, "n": 1}, {"m": 4, "n": 3}]
    )
    def test_points_and_currents_3d(self, cmod_eqdsk, mode):
        """Test get_filament_ds method and create 3D visualization of filament traces."""

        eq_field = EquilibriumField(cmod_eqdsk)
        num_filaments = 7  # Prime: coprime with any n_local

        fig_dir = os.path.join(self.fig_dir, "test_points_and_currents_3d")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        equilibrium_tracer = EquilibriumFilamentTracer(
            mode["m"], mode["n"], eq_field, base_num_points=200
        )

        filament_ds = equilibrium_tracer.get_filament_ds(
            num_filaments=num_filaments, coordinate_system="cartesian"
        )

        # Create 3D plot of filament traces
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        X = filament_ds["x"].values  # Shape: (num_filaments, num_points)
        Y = filament_ds["y"].values  # Shape: (num_filaments, num_points)
        Z = filament_ds["z"].values  # Shape: (num_filaments, num_points)
        currents = filament_ds["current"].values  # Shape: (num_filaments,)

        # Plot each filament with color based on current magnitude
        norm = plt.Normalize(vmin=currents.real.min(), vmax=currents.real.max())
        cmap = plt.cm.viridis

        for i in range(num_filaments):
            color = cmap(norm(currents[i].real))
            ax.plot(X[i, :], Y[i, :], Z[i, :], color=color, linewidth=5, alpha=0.9)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("|I| (Current Real Part)", fontsize=12)

        # Set labels and title
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_zlabel("Z [m]", fontsize=12)
        ax.set_title(
            f"3D Equilibrium Filament Traces (m={mode['m']}, n={mode['n']})",
            fontsize=14,
        )

        # Make the aspect ratio equal on all 3 axes
        max_range = (
            np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
            / 2.0
        )
        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set viewing angle
        ax.view_init(elev=20, azim=45)

        fig.tight_layout()
        fig_path = os.path.join(fig_dir, f"m{mode['m']}_n{mode['n']}_3d.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(
            f"Saved 3D equilibrium filament trace plot for mode m={mode['m']}, n={mode['n']} to {fig_path}"
        )

        # All current magnitudes must be 1 (unit phasors)
        np.testing.assert_allclose(np.abs(currents), 1.0, rtol=1e-10)

        # Dataset shape must match num_filaments x num_points
        assert filament_ds["x"].shape == (num_filaments, equilibrium_tracer.num_points)
        assert filament_ds["y"].shape == (num_filaments, equilibrium_tracer.num_points)
        assert filament_ds["z"].shape == (num_filaments, equilibrium_tracer.num_points)


class TestAllFilamentTracers:
    """Test common functionality for all filament tracers."""

    fig_dir = os.path.join(FIG_DIR, "test_all_filament_tracers")

    @pytest.mark.parametrize(
        "tracer_class, extra_args",
        [
            (ToroidalFilamentTracer, (1.8, 0.0, 0.5, 400)),
            (EquilibriumFilamentTracer, (400,)),
        ],
    )
    @pytest.mark.parametrize(
        "mode", [{"m": 2, "n": 1}, {"m": 3, "n": 2}, {"m": -3, "n": 2}]
    )
    def test_closest_point_has_same_current(
        self, cmod_eqdsk, tracer_class: FilamentTracer, extra_args, mode
    ):
        """For every point, ensure its nearest neighbor has the same current and belongs to the same filament"""
        if tracer_class is EquilibriumFilamentTracer:
            tracer_args = (
                mode["m"],
                mode["n"],
                EquilibriumField(cmod_eqdsk),
            ) + extra_args
        else:
            tracer_args = (mode["m"], mode["n"]) + extra_args
        tracer = tracer_class(*tracer_args)
        filament_ds = tracer.get_filament_ds(
            num_filaments=7, coordinate_system="cartesian"
        )

        x_vals_all = filament_ds["x"].values
        y_vals_all = filament_ds["y"].values
        z_vals_all = filament_ds["z"].values
        num_filaments = filament_ds.sizes["filament"]
        num_pts = filament_ds.sizes["point"]

        # Flatten all filament points to (N_filaments * N_pts, 3) and record which filament each belongs to
        all_points = np.column_stack(
            [x_vals_all.ravel(), y_vals_all.ravel(), z_vals_all.ravel()]
        )
        flat_filament_idx = np.repeat(np.arange(num_filaments), num_pts)

        # Compute the full squared-distance matrix via ||a-b||² = ||a||² + ||b||² - 2·a·b
        norms_sq = np.sum(all_points**2, axis=1)
        dist_sq = (
            norms_sq[:, np.newaxis]
            + norms_sq[np.newaxis, :]
            - 2.0 * (all_points @ all_points.T)
        )
        np.fill_diagonal(dist_sq, np.inf)

        nearest_flat_idx = np.argmin(dist_sq, axis=1)
        nearest_filaments = flat_filament_idx[nearest_flat_idx]

        assert np.all(nearest_filaments == flat_filament_idx), (
            "Some points' nearest neighbor belongs to a different filament. "
            f"Offending point flat indices: {np.where(nearest_filaments != flat_filament_idx)[0].tolist()}"
        )
        assert np.all(
            np.isclose(
                filament_ds["current"].values[nearest_filaments],
                filament_ds["current"].values[flat_filament_idx],
                rtol=1e-10,
            )
        )

    @pytest.mark.parametrize(
        "tracer_class, extra_args",
        [
            (ToroidalFilamentTracer, (1.8, 0.0, 0.5, 400)),
            (EquilibriumFilamentTracer, (400,)),
        ],
    )
    @pytest.mark.parametrize("mode", [{"m": 2, "n": 1}, {"m": 3, "n": 2}])
    def test_helicity_maintains_shape(
        self, cmod_eqdsk, tracer_class: FilamentTracer, extra_args, mode
    ):
        """For each filament tracer, ensure the maximum extents in R and Z space is maintained
        for positive and negative helicity.

        Must ensure the tracer isn't accidentally being shifted up / down in Z when the sign is flipped.
        """
        if tracer_class is EquilibriumFilamentTracer:
            eq_field = EquilibriumField(cmod_eqdsk)
            tracer_args_pos = (mode["m"], mode["n"], eq_field) + extra_args
            tracer_args_neg = (-mode["m"], mode["n"], eq_field) + extra_args
        else:
            tracer_args_pos = (mode["m"], mode["n"]) + extra_args
            tracer_args_neg = (-mode["m"], mode["n"]) + extra_args
        tracer_pos = tracer_class(*tracer_args_pos)
        filament_pos_ds = tracer_pos.get_filament_ds(
            num_filaments=7, coordinate_system="cylindrical"
        )

        tracer_neg = tracer_class(*tracer_args_neg)
        filament_neg_ds = tracer_neg.get_filament_ds(
            num_filaments=7, coordinate_system="cylindrical"
        )

        R_pos = filament_pos_ds["R"].values
        Z_pos = filament_pos_ds["Z"].values
        R_neg = filament_neg_ds["R"].values
        Z_neg = filament_neg_ds["Z"].values

        # Check that the maximum and minimum R and Z values are approximately equal between positive and negative m

        assert np.isclose(R_pos.max(), R_neg.max(), rtol=0.05), (
            f"Maximum R values differ between positive and negative m: {R_pos.max()} vs {R_neg.max()}"
        )
        assert np.isclose(R_pos.min(), R_neg.min(), rtol=0.05), (
            f"Minimum R values differ between positive and negative m: {R_pos.min()} vs {R_neg.min()}"
        )
        assert np.isclose(Z_pos.max(), Z_neg.max(), rtol=0.05), (
            f"Maximum Z values differ between positive and negative m: {Z_pos.max()} vs {Z_neg.max()}"
        )
        assert np.isclose(Z_pos.min(), Z_neg.min(), rtol=0.05), (
            f"Minimum Z values differ between positive and negative m: {Z_pos.min()} vs {Z_neg.min()}"
        )


class TestNegativeMFilament:
    """Tests specifically for negative m handling."""

    @pytest.mark.parametrize(
        "base_num_points, scale_points, prevent_synthetic_structure",
        [
            (100, True, False),
            (100, True, True),
            (100, False, False),
        ],
    )
    def test_toroidal_negative_m_scale_points(
        self, base_num_points, scale_points, prevent_synthetic_structure
    ):
        """ToroidalFilamentTracer with negative m and scale_points must not produce degenerate num_points."""
        m, n = -3, 2
        R0, Z0, a = 1.8, 0.0, 0.5
        tracer = ToroidalFilamentTracer(
            m,
            n,
            R0,
            Z0,
            a,
            base_num_points=base_num_points,
            scale_points=scale_points,
            prevent_synthetic_structure=prevent_synthetic_structure,
        )
        # num_points must be at least as large as it would be for positive m
        assert tracer.num_points >= 2, "num_points must be > 2"
        expected = base_num_points
        if scale_points:
            expected = int(base_num_points * abs(m) / n)
        if prevent_synthetic_structure:
            expected = nextprime(expected)
        assert tracer.num_points == expected

        points, etas = tracer.trace()
        assert points.shape[0] == tracer.num_points
        # phi goes backwards for negative m
        assert points[-1, 1] < 0.0

    def test_negative_m_current_phasor_is_conjugate_of_positive_m(self):
        """m=-3,n=2 must produce complex conjugate currents compared to m=3,n=2."""
        R0, Z0, a = 1.8, 0.0, 0.5
        num_filaments = 7

        pos_tracer = ToroidalFilamentTracer(3, 2, R0, Z0, a, base_num_points=100)
        neg_tracer = ToroidalFilamentTracer(-3, 2, R0, Z0, a, base_num_points=100)

        pos_ds = pos_tracer.get_filament_ds(num_filaments=num_filaments)
        neg_ds = neg_tracer.get_filament_ds(num_filaments=num_filaments)

        np.testing.assert_allclose(
            neg_ds["current"].values,
            np.conj(pos_ds["current"].values),
            atol=1e-10,
        )

    @pytest.mark.xfail(
        reason="Helicity sign is presently not treated well, will fix in COCOS PR"
    )
    def test_helicity_sign_mirrors_z_coordinates(self, cmod_eqdsk):
        """EquilibriumFilamentTracer with helicity_sign=+1 vs -1 must produce Z-mirrored traces."""
        eq_field = EquilibriumField(cmod_eqdsk)
        m, n = 3, 2

        tracer_pos = EquilibriumFilamentTracer(
            m,
            n,
            eq_field,
            base_num_points=101,
            scale_points=False,
            prevent_synthetic_structure=False,
            helicity_sign=1,
        )
        tracer_neg = EquilibriumFilamentTracer(
            m,
            n,
            eq_field,
            base_num_points=101,
            scale_points=False,
            prevent_synthetic_structure=False,
            helicity_sign=-1,
        )

        pts_pos, _ = tracer_pos.trace()
        pts_neg, _ = tracer_neg.trace()

        # Z displacement about the magnetic axis should be negated between the two tracers.
        # Tolerance is loose because the equilibrium is not perfectly up-down symmetric,
        # so a_pos(eta) != a_neg(eta) and the mirror is approximate.
        # Similarly, R is not necessarily equal for a non-up-down-symmetric equilibrium.
        zmagx = eq_field.eqdsk.zmagx
        np.testing.assert_allclose(
            pts_pos[:, 2] - zmagx,
            -(pts_neg[:, 2] - zmagx),
            rtol=0.05,
            err_msg="helicity_sign=+1 and -1 must produce Z-mirrored traces about the magnetic axis",
        )
        np.testing.assert_allclose(
            pts_pos[:, 0],
            pts_neg[:, 0],
            rtol=0.05,
            err_msg="helicity_sign should not affect R coordinates significantly",
        )


class TestFilamentValueErrors:
    """Tests for ValueError conditions in get_filament_ds / get_filament_list."""

    def test_get_filament_ds_zero_filaments_raises(self):
        tracer = ToroidalFilamentTracer(
            2,
            1,
            1.8,
            0.0,
            0.5,
            base_num_points=50,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        with pytest.raises(
            ValueError, match="num_filaments must be a positive integer"
        ):
            tracer.get_filament_ds(num_filaments=0)

    def test_get_filament_ds_negative_filaments_raises(self):
        tracer = ToroidalFilamentTracer(
            2,
            1,
            1.8,
            0.0,
            0.5,
            base_num_points=50,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        with pytest.raises(
            ValueError, match="num_filaments must be a positive integer"
        ):
            tracer.get_filament_ds(num_filaments=-1)

    def test_get_filament_ds_invalid_coordinate_system_raises(self):
        tracer = ToroidalFilamentTracer(
            2,
            1,
            1.8,
            0.0,
            0.5,
            base_num_points=50,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        with pytest.raises(ValueError, match="coordinate_system"):
            tracer.get_filament_ds(num_filaments=7, coordinate_system="spherical")

    def test_get_filament_ds_non_coprime_num_filaments_raises(self):
        """num_filaments not coprime with n_local must raise ValueError."""
        # m=3, n=2: n_local=2. num_filaments=4: gcd(4,2)=2 != 1
        tracer = ToroidalFilamentTracer(
            3,
            2,
            1.8,
            0.0,
            0.5,
            base_num_points=50,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        with pytest.raises(ValueError, match="coprime"):
            tracer.get_filament_ds(num_filaments=4)

    def test_get_filament_list_zero_filaments_raises(self):
        tracer = ToroidalFilamentTracer(
            2,
            1,
            1.8,
            0.0,
            0.5,
            base_num_points=50,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        with pytest.raises(
            ValueError, match="num_filaments must be a positive integer"
        ):
            tracer.get_filament_list(num_filaments=0)

    def test_get_filament_list_invalid_coordinate_system_raises(self):
        tracer = ToroidalFilamentTracer(
            2,
            1,
            1.8,
            0.0,
            0.5,
            base_num_points=50,
            scale_points=False,
            prevent_synthetic_structure=False,
        )
        with pytest.raises(ValueError, match="coordinate_system"):
            tracer.get_filament_list(num_filaments=7, coordinate_system="toroidal")
