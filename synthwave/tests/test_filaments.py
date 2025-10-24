import pytest
import numpy as np

from synthwave.magnetic_geometry.filaments import ToroidalFilamentTracer, EquilibriumFilamentTracer

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
    def test_trace_circular_geometry(self, R0, Z0, a):
        """Test that the filament traces a circle in the poloidal plane."""
        m, n = 1, 1
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
        expected_phi_range = 2 * np.pi * m
        np.testing.assert_allclose(phi_range, expected_phi_range, rtol=1e-10)

        np.testing.assert_allclose(etas, np.linspace(0, 2 * np.pi, num_points), rtol=1e-10)
 
    def test_trace_mode_scaling(self):
        """Test that phi scales correctly with mode number m."""
        R0, Z0, a = 1.5, 0.0, 0.4
        num_points = 50
        
        # Test different mode numbers
        for m in [1, 2, 3]:
            n = 1
            tracer = ToroidalFilamentTracer(m, n, R0, Z0, a, num_points)
            points, etas = tracer.trace()

            phi = points[:, 1]
            phi_range = phi[-1] - phi[0]
            expected_phi_range = 2 * np.pi * m
            
            np.testing.assert_allclose(phi_range, expected_phi_range, rtol=1e-10,
                                     err_msg=f"Failed for m={m}")
    
    def test_trace_specific_points(self):
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
        # At 1/4 of the way around poloidal angle (π/2)
        quarter_idx = 2
        expected_R_quarter = R0  # At top of circle
        expected_Z_quarter = Z0 + a
        np.testing.assert_allclose(R[quarter_idx], expected_R_quarter, atol=1e-10)
        np.testing.assert_allclose(Z[quarter_idx], expected_Z_quarter, atol=1e-10)

        # At 1/2 of the way around poloidal angle (π) 
        half_idx = 4
        expected_R_half = R0 - a  # At inboard midplane
        expected_Z_half = Z0
        np.testing.assert_allclose(R[half_idx], expected_R_half, atol=1e-10)
        np.testing.assert_allclose(Z[half_idx], expected_Z_half, atol=1e-10)

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
    
    @pytest.mark.parametrize("m,n", [(1, 1), (2, 1), (3, 2), (4, 3)])
    def test_trace_various_modes(self, m, n):
        """Test tracing for various mode numbers."""
        R0, Z0, a = 1.7, 0.1, 0.4
        num_points = 100

        tracer = ToroidalFilamentTracer(m, n, R0, Z0, a, num_points)
        points, etas = tracer.trace()

        R, phi, Z = points[:, 0], points[:, 1], points[:, 2]
        
        # Check basic geometry properties
        distances = np.sqrt((R - R0)**2 + (Z - Z0)**2)
        np.testing.assert_allclose(distances, a, rtol=1e-10)
        
        # Check phi range
        phi_range = phi[-1] - phi[0]
        expected_phi_range = 2 * np.pi * m
        np.testing.assert_allclose(phi_range, expected_phi_range, rtol=1e-10)
        
        # Check that all coordinates are finite
        assert np.all(np.isfinite(R))
        assert np.all(np.isfinite(phi))
        assert np.all(np.isfinite(Z))
