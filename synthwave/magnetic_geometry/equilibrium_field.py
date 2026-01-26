import numpy as np

from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline, make_interp_spline
from scipy.optimize import newton

from synthwave.magnetic_geometry.utils import (
    cylindrical_to_cartesian,
    cartesian_to_cylindrical,
)


def biot_savart_cartesian(
    eval_point: np.ndarray, filament_points: np.ndarray, filament_current: float
):
    """Biot-savart law in cartesian coordinates."""
    r_prime = eval_point - filament_points
    r_prime_norm = np.linalg.norm(r_prime, axis=1)
    dl = np.gradient(filament_points, axis=0)
    dl_cross_r = np.cross(dl, r_prime)
    B_segments = (
        (mu_0 / (4 * np.pi))
        * filament_current
        * dl_cross_r
        / (r_prime_norm[:, np.newaxis] ** 3)
    )
    B_total = np.sum(B_segments, axis=0)
    return B_total


def biot_savart_cylindrical(
    eval_point: np.ndarray, filament_points: np.ndarray, filament_current: float
):
    """Biot-savart law in cylindrical coordinates."""
    eval_point = cylindrical_to_cartesian(*eval_point)
    filament_points_cartesian = cylindrical_to_cartesian(*filament_points.T).T
    B_cartesian = biot_savart_cartesian(
        eval_point, filament_points_cartesian, filament_current
    )
    B_cylindrical = cartesian_to_cylindrical(*B_cartesian)
    return B_cylindrical


class EquilibriumField:
    def __init__(self, eqdsk, lam=1e-7):
        self.eqdsk = eqdsk
        self.psi = RectBivariateSpline(
            eqdsk.r_grid[:, 0], eqdsk.z_grid[0, :], eqdsk.psi,
            kx=3, ky=3, s=0
        )

        # Linear grid of psi for 1D profiles
        # https://freeqdsk.readthedocs.io/en/stable/geqdsk.html
        self.psi_grid = np.linspace(eqdsk.simagx, eqdsk.sibdry, eqdsk.nx)
        
        # q(psi)
        # RNC EDIT: Switching to smoothing spline to avoid strange "discretization" jumps
        # Note: lam value (lower = less smoothing) should be reasonably consistent across q 
        # profiles, but this is not certain. Lam=1e-7 works for q(psi) and F(psi) so far.
        self.qpsi = make_smoothing_spline(
            self.psi_grid, eqdsk.qpsi, lam=lam, axis=0
        )

        # F(psi)
        self.F = make_smoothing_spline(
            self.psi_grid, eqdsk.fpol, lam=lam, axis=0
        )

    def get_field_at_point(self, R, Z) -> np.ndarray:
        # Bp = Br + Bz = (d(psi)/dZ - d(psi)/dR) / R
        psir = self.psi.ev(R, Z, dx=1, dy=0)
        psiz = self.psi.ev(R, Z, dx=0, dy=1)

        Br = psiz / R
        Bz = -psir / R
        Bt = self.F(self.psi.ev(R, Z)) / R

        return np.array([Br, Bt, Bz])

    def get_psi_of_q(self, q):
        """Get psi corresponding to a given q"""
        qpsi_grid = self.qpsi(self.psi_grid)
        psi_guess = self.psi_grid[np.argmin(np.abs(qpsi_grid - q))]
        psi = newton(
            func=lambda psi: self.qpsi(psi) - q,
            x0=psi_guess,
            fprime=lambda psi: self.qpsi.derivative(1)(psi),
            maxiter=400, tol=1e-3
        )

        # RNC EDIT:
        # CHECK: For q~<=1, depening on the resolution of the gEQDSK file, 
        # the interpolation function can request a psi value at or less 
        # than the minimum in the psi_grid vector
        # Check to see if the value we want plausibly exists in the final set
        # of q values from the eqdsk
        # The issue appears to be that although psi is continuous and monotonic,
        # q values are "grouped" in an odd, stepwise fashion
        if psi <= self.psi_grid[0]:
            # check if there's a range of possible q-values (e.g.) if this is plausably a resolution issue
            tol = 1e-3
            if np.argwhere(qpsi_grid>(qpsi_grid[0]+1e-3)).squeeze()[0] > 1: # multiple identical q values in a row
                lin_interp_q = np.polyfit(self.psi_grid[:30],qpsi_grid[:30], 1)
                psi = self.psi_grid[ np.argmin(np.abs(np.polyval(lin_interp_q,self.psi_grid[:30]) - q)) ]
            
            if psi > self.psi_grid[0]:
                return psi
            else:
                raise SyntaxError(
                    'Error: requested q=%1.3f is outside the gEQDSK range (q_min = %1.3f)'
                    % (q, qpsi_grid[0])
                )
        return psi