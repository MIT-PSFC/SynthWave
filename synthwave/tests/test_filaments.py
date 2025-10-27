import pytest
import numpy as np
import os
import tempfile

from synthwave import PACKAGE_ROOT

from synthwave.magnetic_geometry.filaments import ToroidalFilamentTracer, EquilibriumFilamentTracer
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(PACKAGE_ROOT, "tests", "figures")

class TestToroidalFilament:
    """Test the CylindricalFilament class."""
    
    def test_init(self):
        """Test initialization of CylindricalFilament."""
        m, n = 2, 1
        R0, Z0, a = 1.8, 0.0, 0.5
        num_points = 100

        tracer = ToroidalFilamentTracer(m, n, R0, Z0, a, num_points)

        assert tracer.m == m
        assert tracer.n == n
        assert tracer.R0 == R0
        assert tracer.Z0 == Z0
        assert tracer.a == a
        assert tracer.num_points == num_points

    def test_trace_basic(self):
        """Test basic tracing functionality."""
        m, n = 2, 1
        R0, Z0, a = 1.8, 0.0, 0.5
        num_points = 100

        filament = ToroidalFilamentTracer(m, n, R0, Z0, a, num_points)
        points, etas = filament.trace()
        
        # Check shape
        assert points.shape == (num_points, 3)
        assert etas.shape == (num_points,)
    
    @pytest.mark.parametrize("R0", [1.5, 2.0, 2.5])
    @pytest.mark.parametrize("Z0", [0.0, 0.1, -0.1])
    @pytest.mark.parametrize("a", [0.4, 0.5, 0.6])
    @pytest.mark.parametrize("mode", [(1, 1), (2, 1), (3, 2), (3, 1)])
    def test_trace_circular_geometry(self, R0, Z0, a, mode):
        """Test that the filament traces a circle in the poloidal plane."""
        m, n = mode
        num_points = 100

        tracer = ToroidalFilamentTracer(m, n, R0, Z0, a, num_points)
        points, etas = tracer.trace()

        R, phi, Z = points[:, 0], points[:, 1], points[:, 2]
        
        # Check that points form a circle in R-Z plane
        # Distance from magnetic axis should be constant (= minor radius)
        distances = np.sqrt((R - R0)**2 + (Z - Z0)**2)
        np.testing.assert_allclose(distances, a, rtol=1e-10)
        
        # Check that we complete the correct number of toroidal turns
        phi_range = phi[-1] - phi[0]
        expected_phi_range = 2 * np.pi * m / n
        np.testing.assert_allclose(phi_range, expected_phi_range, rtol=1e-10)

        np.testing.assert_allclose(etas, np.linspace(0, 2 * np.pi, num_points), rtol=1e-10)

    def test_trace_specific_points_2_1(self):
        """Test that specific points are at expected locations."""
        m, n = 2, 1
        R0, Z0, a = 1.8, 0.0, 0.5
        num_points = 9  # Use 9 points for easy checking. This gives points at 0, π/4, π/2, ..., 2π
        
        tracer = ToroidalFilamentTracer(m, n, R0, Z0, a, num_points)
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
        
        tracer = ToroidalFilamentTracer(m, n, R0, Z0, a, num_points)
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

        tracer = ToroidalFilamentTracer(m, n, R0, Z0, a, num_points=100)

        # Test with default number of points
        points_default, _ = tracer.trace()
        assert points_default.shape[0] == 100
        
        # Test with custom number of points
        points_custom, _ = tracer.trace(num_filament_points=200)
        assert points_custom.shape[0] == 200
        
        # Both should have same geometry, just different resolution
        R_default, phi_default, Z_default = points_default[:, 0], points_default[:, 1], points_default[:, 2]
        R_custom, phi_custom, Z_custom = points_custom[:, 0], points_custom[:, 1], points_custom[:, 2]
        
        # Check that distances from axis are still correct
        distances_default = np.sqrt((R_default - R0)**2 + (Z_default - Z0)**2)
        distances_custom = np.sqrt((R_custom - R0)**2 + (Z_custom - Z0)**2)
        
        np.testing.assert_allclose(distances_default, a, rtol=1e-10)
        np.testing.assert_allclose(distances_custom, a, rtol=1e-10)

    @pytest.mark.parametrize("mode", [{"m": 1, "n": 1}, {"m": 2, "n": 1}, {"m": 3, "n": 2}, {"m": 3, "n": 1}, {"m": 4, "n": 3}, {"m": 5, "n": 4}])
    def test_points_and_currents(self, mode):
        """Test get_filament_list method for correct output dataset."""
        major_radius = 1
        minor_radius = 0.3
        num_filaments = 24

        fig_dir = os.path.join(FIG_DIR, "test_points_and_currents")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        toroidal_tracer = ToroidalFilamentTracer(
            mode["m"], mode["n"], major_radius, 0, minor_radius, num_points=100
        )

        filament_ds = toroidal_tracer.get_filament_ds(num_filaments=num_filaments, coordinate_system="toroidal")

        # Create plot of filament points in phi and eta coordinates
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Extract data
        phi_vals = filament_ds["phi"].values  # Shape: (num_filaments, num_points)
        eta_vals = filament_ds["eta"].values  # Shape: (num_points,)
        currents = filament_ds["current"].values  # Shape: (num_filaments,), complex values

        # Determine the phi range for appropriate x-axis ticks
        phi_min = phi_vals.min()
        phi_max = phi_vals.max()

        # Create x-axis ticks that span the entire phi domain
        # Number of ticks based on the range (use pi/2 increments)
        num_x_ticks = int(np.ceil(phi_max / (np.pi/2))) + 1
        x_ticks = [i * np.pi/2 for i in range(num_x_ticks)]
        x_labels = []
        for tick in x_ticks:
            # Convert to fraction of pi
            ratio = tick / np.pi
            if ratio == 0:
                x_labels.append('0')
            elif ratio == 0.5:
                x_labels.append(r'$\pi/2$')
            elif ratio == 1:
                x_labels.append(r'$\pi$')
            elif ratio == 1.5:
                x_labels.append(r'$3\pi/2$')
            elif ratio == 2:
                x_labels.append(r'$2\pi$')
            elif ratio == 2.5:
                x_labels.append(r'$5\pi/2$')
            elif ratio == 3:
                x_labels.append(r'$3\pi$')
            elif ratio == 3.5:
                x_labels.append(r'$7\pi/2$')
            elif ratio == 4:
                x_labels.append(r'$4\pi$')
            else:
                # For other values, format as fraction
                from fractions import Fraction
                frac = Fraction(int(ratio * 2), 2).limit_denominator()
                if frac.denominator == 1:
                    x_labels.append(f'${frac.numerator}\pi$')
                else:
                    x_labels.append(f'${frac.numerator}\pi/{frac.denominator}$')

        # Plot real component of current
        scatter1 = ax1.scatter(
            phi_vals.flatten(),
            np.tile(eta_vals, num_filaments),
            c=np.repeat(currents.real, phi_vals.shape[1]),
            cmap='RdBu_r',
            s=20,
            alpha=0.6
        )
        ax1.set_xlabel('phi')
        ax1.set_ylabel('eta')
        ax1.set_title(f'Real Component of Current (m={mode["m"]}, n={mode["n"]})')
        ax1.grid(True, alpha=0.3)

        # Set x-axis ticks with pi fractions
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels)

        # Set y-axis ticks with pi fractions
        ax1.set_yticks([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi,
                        7*np.pi/6, 5*np.pi/4, 4*np.pi/3, 3*np.pi/2, 5*np.pi/3, 7*np.pi/4, 11*np.pi/6, 2*np.pi])
        ax1.set_yticklabels(['0', r'$\pi/6$', r'$\pi/4$', r'$\pi/3$', r'$\pi/2$', r'$2\pi/3$', r'$3\pi/4$',
                                r'$5\pi/6$', r'$\pi$', r'$7\pi/6$', r'$5\pi/4$', r'$4\pi/3$', r'$3\pi/2$',
                                r'$5\pi/3$', r'$7\pi/4$', r'$11\pi/6$', r'$2\pi$'])

        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Re(I)')

        # Plot imaginary component of current
        scatter2 = ax2.scatter(
            phi_vals.flatten(),
            np.tile(eta_vals, num_filaments),
            c=np.repeat(currents.imag, phi_vals.shape[1]),
            cmap='RdBu_r',
            s=20,
            alpha=0.6
        )
        ax2.set_xlabel('phi')
        ax2.set_ylabel('eta')
        ax2.set_title(f'Imaginary Component of Current (m={mode["m"]}, n={mode["n"]})')
        ax2.grid(True, alpha=0.3)

        # Set x-axis ticks with pi fractions (same as ax1)
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels)

        # Set y-axis ticks with pi fractions
        ax2.set_yticks([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi,
                        7*np.pi/6, 5*np.pi/4, 4*np.pi/3, 3*np.pi/2, 5*np.pi/3, 7*np.pi/4, 11*np.pi/6, 2*np.pi])
        ax2.set_yticklabels(['0', r'$\pi/6$', r'$\pi/4$', r'$\pi/3$', r'$\pi/2$', r'$2\pi/3$', r'$3\pi/4$',
                                r'$5\pi/6$', r'$\pi$', r'$7\pi/6$', r'$5\pi/4$', r'$4\pi/3$', r'$3\pi/2$',
                                r'$5\pi/3$', r'$7\pi/4$', r'$11\pi/6$', r'$2\pi$'])

        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Im(I)')

        fig.tight_layout()
        fig_path = os.path.join(fig_dir, f'm{mode["m"]}_n{mode["n"]}.png')  
        fig.savefig(fig_path, dpi=150)
        print(f'Saved filament current plot for mode m={mode["m"]}, n={mode["n"]} to {fig_path}')

            
