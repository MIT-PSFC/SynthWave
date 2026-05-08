import numpy as np
from OpenFUSIONToolkit.ThinCurr.meshing import ThinCurr_periodic_toroid
from sympy import nextprime


def cylindrical_to_cartesian(R: float, phi: float, Z: float) -> np.ndarray:
    """Convert cylindrical coordinates (R, phi, Z) to Cartesian coordinates (x, y, z)."""
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    return np.array([x, y, Z])


def cartesian_to_cylindrical(x: float, y: float, z: float) -> np.ndarray:
    """Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (R, phi, Z)."""
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([R, phi, z])


def phi_domain(phi: np.ndarray) -> np.ndarray:
    """Ensure phi is in the range [0, 2*pi)"""
    return np.mod(phi, 2 * np.pi)


def angle_domain(phase: np.ndarray) -> np.ndarray:
    """Ensure phase is within (-pi, pi],
    to match np.angle output when determining the phase of a complex number"""
    return np.angle(np.exp(1j * phase))


def wrapped_diff(phase1: np.ndarray, phase2: np.ndarray) -> np.ndarray:
    """Compute the wrapped difference between two phases.
    For example, if phase1 is 3pi / 4 and phase2 is -3pi / 4,
    going from phase1 to phase2 is a change of 3 pi / 2,
    which is equivalent to -pi / 2 when wrapped to the range (-pi, pi].
    """
    diff = phase1 - phase2
    return angle_domain(diff)


def create_torus_mesh(R0, a, ntheta=64, nphi=128):
    # Create r_grid: [nphi, ntheta, 3] array defining the surface of one field period
    nfp = 1
    # I want these to be as high resolution as possible without segfaulting ThinCurr
    ntheta = nextprime(ntheta)  # Example was originally 40
    nphi = nextprime(nphi)  # Example was originally 80

    # Create poloidal and toroidal angle grids
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)

    # Create meshgrid
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")

    # Calculate Cartesian coordinates for the torus surface
    # r_grid shape: [nphi, ntheta, 3]
    r_grid = np.zeros((nphi, ntheta, 3))
    r_grid[:, :, 0] = (R0 + a * np.cos(theta_grid)) * np.cos(phi_grid)  # x
    r_grid[:, :, 1] = (R0 + a * np.cos(theta_grid)) * np.sin(phi_grid)  # y
    r_grid[:, :, 2] = a * np.sin(theta_grid)  # z

    torus_mesh = ThinCurr_periodic_toroid(r_grid, nfp, ntheta, nphi)
    return torus_mesh
