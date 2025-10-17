import numpy as np


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
