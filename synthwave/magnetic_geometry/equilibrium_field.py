import numpy as np
from freeqdsk.geqdsk import GEQDSKFile
from loguru import logger
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline, make_smoothing_spline
from scipy.optimize import newton

from synthwave.magnetic_geometry.utils import (
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
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


def detect_cocos(eqdsk: GEQDSKFile):
    """Detect the COCOS of a given GEQDSK file
    See `Sauter et al, 2013 <https://doi.org/10.1016/j.cpc.2012.09.010>`_.
    Also https://crppwww.epfl.ch/~sauter/cocos/ and https://crppwww.epfl.ch/~sauter/cocos/Sauter_COORD_CONVENTIONS_COCOS_2012_updated_after_reprint_for_Appendices_and_refs.pdf

    COCOS is defined by the following:
    - e_Bp: 0 if psi in Wb/rad, 1 if psi in Weber
    - sign_Bp: sign of poloidal magnetic field (Bp)
    - sign_RphiZ: +1 if (R, phi, Z), -1 if (R, Z, phi)
    - sign_rhotp: +1 if (rho, theta, phi), -1 if (rho, phi, theta)

    sign_rhotp is derived from sign(Ip * B0) (discharge helicity) rather than sign(q),
    because some codes store abs(q). From Sauter Table I: sign(q) = sign_Bp * sign_RphiZ * sign_rhotp.
    For sign_RphiZ=+1 codes, sign(B0) = sign(q_physical), making this equivalent to the
    Sauter criterion. For sign_RphiZ=-1 codes, sign(B0) gives consistent results when B0
    has its physical sign.
    """
    sign_Ip = np.sign(float(eqdsk.cpasma))
    sign_B0 = np.sign(float(eqdsk.bcentr))
    psi_increasing = float(np.sign(eqdsk.sibdry - eqdsk.simagx))
    # From table III: sign(dpsi) = sign_Bp * sign_Ip
    sign_Bp = int(psi_increasing * sign_Ip)

    # sign_RphiZ = +1 (R,phi,Z, phi CCW): F = R*B_phi has same sign as B0.
    # sign_RphiZ = -1 (R,Z,phi, phi CW): stored F has opposite sign to physical B0.
    # When bcentr=0 (not stored), assume standard sign_RphiZ=+1 (gEQDSK default).
    sign_fpol = int(np.sign(np.nanmean(eqdsk.fpol)))
    if sign_B0 != 0:
        sign_RphiZ = sign_fpol * int(sign_B0)
    else:
        logger.warning(
            "bcentr=0, unable to determine sign_RphiZ from F. Assuming +1 (gEQDSK default)."
        )
        sign_RphiZ = 1

    def _e_Bp(eqdsk):
        # Detect e_Bp via Grad-Shafranov residual.
        # The GS equation for psi in Wb/rad (e_Bp=0) is:
        #   Delta*(psi) = -(mu_0*R^2*pprime + ffprime)
        # If psi is in Weber (e_Bp=1) the same stored pprime/ffprime satisfy:
        #   Delta*(psi) = -(2*pi)^2 * (mu_0*R^2*pprime + ffprime)
        # Fit alpha such that lhs = alpha * rhs_ebp0: alpha~1 -> e_Bp=0, alpha~(2*pi)^2 -> e_Bp=1.
        R_1d = np.array(eqdsk.r_grid[:, 0], dtype=float)
        Z_1d = np.array(eqdsk.z_grid[0, :], dtype=float)
        psi_2d = np.array(eqdsk.psi, dtype=float)
        if psi_2d.shape == (len(Z_1d), len(R_1d)):
            psi_2d = psi_2d.T  # ensure (nR, nZ)

        dpsi_dR = np.gradient(psi_2d, R_1d, axis=0)
        lhs_gs = (
            np.gradient(dpsi_dR, R_1d, axis=0)
            - dpsi_dR / R_1d[:, None]
            + np.gradient(np.gradient(psi_2d, Z_1d, axis=1), Z_1d, axis=1)
        )

        psi_norm_2d = np.clip(
            (psi_2d - float(eqdsk.simagx))
            / (float(eqdsk.sibdry) - float(eqdsk.simagx)),
            0.0,
            1.0,
        )
        pprime_raw = np.array(eqdsk.pprime, dtype=float)
        ffprime_raw = np.array(eqdsk.ffprime, dtype=float)
        psi_norm_1d = np.linspace(0.0, 1.0, len(pprime_raw))
        pprime_2d = np.interp(psi_norm_2d, psi_norm_1d, pprime_raw)
        ffprime_2d = np.interp(psi_norm_2d, psi_norm_1d, ffprime_raw)

        rhs_gs = -(mu_0 * R_1d[:, None] ** 2 * pprime_2d + ffprime_2d)

        sl = np.s_[3:-3, 3:-3]
        lhs_flat = lhs_gs[sl].ravel()
        rhs_flat = rhs_gs[sl].ravel()
        mask = np.abs(rhs_flat) > 1e-6 * np.max(np.abs(rhs_flat))
        alpha = np.dot(lhs_flat[mask], rhs_flat[mask]) / np.dot(
            rhs_flat[mask], rhs_flat[mask]
        )
        e_Bp = 0 if abs(alpha - 1.0) < abs(alpha - (2 * np.pi) ** 2) else 1

        return e_Bp

    e_Bp = _e_Bp(eqdsk)

    # sign_rhotp is the discharge helicity: sign(Ip * B0).
    # Using sign(q) is unreliable because some codes store abs(q).
    # When bcentr=0, infer sign_B0 from fpol under the standard sign_RphiZ=+1 assumption:
    # F = R*B_phi, so sign_fpol = sign_RphiZ * sign_B0 = sign_B0 when sign_RphiZ=+1.
    if sign_B0 != 0:
        sign_B0_eff = int(sign_B0)
    else:
        sign_B0_eff = int(sign_fpol)
        logger.warning(
            "bcentr=0, unable to determine sign_B0. Inferring sign_B0=%d from F under the standard sign_RphiZ=+1 assumption."
        )
    sign_rhotp = int(sign_Ip) * sign_B0_eff

    # From Table I
    # (e_Bp, sign_Bp, sign_RphiZ, sign_rhotp) -> COCOS number
    cocos_lookup = {
        (0, +1, +1, +1): 1,
        (1, +1, +1, +1): 11,
        (0, +1, -1, +1): 2,
        (1, +1, -1, +1): 12,
        (0, -1, +1, -1): 3,
        (1, -1, +1, -1): 13,
        (0, -1, -1, -1): 4,
        (1, -1, -1, -1): 14,
        (0, +1, +1, -1): 5,
        (1, +1, +1, -1): 15,
        (0, +1, -1, -1): 6,
        (1, +1, -1, -1): 16,
        (0, -1, +1, +1): 7,
        (1, -1, +1, +1): 17,
        (0, -1, -1, +1): 8,
        (1, -1, -1, +1): 18,
    }
    cocos_input = cocos_lookup.get((e_Bp, sign_Bp, sign_RphiZ, sign_rhotp), None)

    if cocos_input is None:
        raise ValueError(
            "Could not determine COCOS for the given GEQDSK. Please check the signs of Bp, RphiZ, and rhotp."
        )

    return cocos_input


def convert_cocos(
    eqdsk: GEQDSKFile, cocos_target: int, cocos_input: int | None = None
) -> GEQDSKFile:
    """Convert a GEQDSKFile from freeqdsk to the target COCOS
    See `Sauter et al, 2013 <https://doi.org/10.1016/j.cpc.2012.09.010>`_.
    Also https://crppwww.epfl.ch/~sauter/cocos/ and https://crppwww.epfl.ch/~sauter/cocos/Sauter_COORD_CONVENTIONS_COCOS_2012_updated_after_reprint_for_Appendices_and_refs.pdf

    Args:
        eqdsk: GEQDSKFile object from freeqdsk
        cocos_target: COCOS to convert to (1-8, 11-18)
        cocos_input: COCOS of the input GEQDSKFile (if None, it will be determined automatically)

    Returns:
        A new GEQDSKFile object with the specified COCOS
    """

    if cocos_target not in [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]:
        raise ValueError("Invalid target COCOS: %d" % cocos_target)

    if cocos_input is None:
        cocos_input = detect_cocos(eqdsk)

    if cocos_input == cocos_target:
        logger.debug("No conversion needed, already in COCOS %d" % cocos_target)
    else:
        logger.debug(
            "Converting from COCOS %d to COCOS %d" % (cocos_input, cocos_target)
        )

    # COCOS number -> (e_Bp, sign_Bp, sign_RphiZ, sign_rhotp)
    cocos_params = {
        1: (0, +1, +1, +1),
        11: (1, +1, +1, +1),
        2: (0, +1, -1, +1),
        12: (1, +1, -1, +1),
        3: (0, -1, +1, -1),
        13: (1, -1, +1, -1),
        4: (0, -1, -1, -1),
        14: (1, -1, -1, -1),
        5: (0, +1, +1, -1),
        15: (1, +1, +1, -1),
        6: (0, +1, -1, -1),
        16: (1, +1, -1, -1),
        7: (0, -1, +1, +1),
        17: (1, -1, +1, +1),
        8: (0, -1, -1, +1),
        18: (1, -1, -1, +1),
    }
    e_Bp_i, sign_Bp_i, sign_RphiZ_i, sign_rhotp_i = cocos_params[cocos_input]
    e_Bp_o, sign_Bp_o, sign_RphiZ_o, sign_rhotp_o = cocos_params[cocos_target]

    # Conversion factors from Sauter 2013, Table III
    psi_factor = (sign_Bp_o / sign_Bp_i) * (2 * np.pi) ** (e_Bp_o - e_Bp_i)
    F_factor = sign_RphiZ_o / sign_RphiZ_i
    q_factor = (sign_rhotp_o * sign_Bp_o * sign_RphiZ_o) / (
        sign_rhotp_i * sign_Bp_i * sign_RphiZ_i
    )
    # pprime = dp/dpsi; ffprime = F*dF/dpsi -> both scale as 1/psi_factor
    # (F_factor^2 = 1 always, so the F sign cancels in ffprime)

    simagx = float(eqdsk.simagx) * psi_factor
    sibdry = float(eqdsk.sibdry) * psi_factor
    psi = np.array(eqdsk.psi, dtype=float) * psi_factor

    new_eqdsk = GEQDSKFile(
        # Unchanged
        comment=eqdsk.comment,
        shot=eqdsk.shot,
        nx=eqdsk.nx,
        ny=eqdsk.ny,
        rdim=eqdsk.rdim,
        zdim=eqdsk.zdim,
        rcentr=eqdsk.rcentr,
        rleft=eqdsk.rleft,
        zmid=eqdsk.zmid,
        rmagx=eqdsk.rmagx,
        zmagx=eqdsk.zmagx,
        bcentr=eqdsk.bcentr,
        cpasma=eqdsk.cpasma,
        pres=eqdsk.pres,
        nbdry=eqdsk.nbdry,
        nlim=eqdsk.nlim,
        rbdry=eqdsk.rbdry,
        zbdry=eqdsk.zbdry,
        rlim=eqdsk.rlim,
        zlim=eqdsk.zlim,
        # COCOS-dependent
        simagx=simagx,
        sibdry=sibdry,
        psi=psi,
        fpol=np.array(eqdsk.fpol, dtype=float) * F_factor,
        ffprime=np.array(eqdsk.ffprime, dtype=float) / psi_factor,
        pprime=np.array(eqdsk.pprime, dtype=float) / psi_factor,
        qpsi=np.array(eqdsk.qpsi, dtype=float) * q_factor,
    )

    return new_eqdsk


class EquilibriumField:
    def __init__(self, eqdsk, lam=1e-7):
        eqdsk = convert_cocos(eqdsk, cocos_target=1)  # Convert to COCOS 1 internally

        self.eqdsk = eqdsk
        self.psi = RectBivariateSpline(
            eqdsk.r_grid[:, 0], eqdsk.z_grid[0, :], eqdsk.psi, kx=3, ky=3, s=0
        )

        # Linear grid of psi for 1D profiles
        # https://freeqdsk.readthedocs.io/en/stable/geqdsk.html
        self.psi_grid = np.linspace(eqdsk.simagx, eqdsk.sibdry, eqdsk.nx)

        # Switching to smoothing spline to avoid strange "discretization" jumps
        # Note: lam value (lower = less smoothing) should be reasonably consistent across q
        # profiles, but this is not certain. Lam=1e-7 works for q(psi) and F(psi) so far.

        # q(psi)
        self.qpsi = make_smoothing_spline(self.psi_grid, eqdsk.qpsi, lam=lam, axis=0)
        # F(psi)
        self.F = make_smoothing_spline(self.psi_grid, eqdsk.fpol, lam=lam, axis=0)

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
            maxiter=400,
            tol=1e-3,
        )

        # psi
        psi = self.core_psi_consistency_check(qpsi_grid, psi, q)

        return psi

    def core_psi_consistency_check(self, qpsi_grid, psi, q):
        """Check for q~<=1
        Depending on the resolution of the gEQDSK file, the interpolation function can request
        a psi value at or less than the minimum in the psi_grid vector
        Check to see if the value we want plausibly exists in the final set of q values from the eqdsk
        The issue appears to be that although psi is continuous and monotonic,
        q values can be "grouped" in an odd, stepwise fashion

        Note that this treats
        """
        # For COCOS 1, psi is positive-monotonic: index 0 at axis (low psi), index -1 at LCFS (high psi)
        # Also using |q| instead of signed q to avoid issues with sign flips.
        qpsi_grid = np.abs(qpsi_grid)

        # If psi is in range, return it unchanged
        if (psi >= self.psi_grid[0]) and (psi <= self.psi_grid[-1]):
            return psi

        # If psi is too large, already lost
        if psi > self.psi_grid[-1]:
            raise ValueError(
                "Error: requested q=%1.3f is outside the gEQDSK range (q_max = %1.3f) and was unable to be fixed"
                % (q, qpsi_grid[-1])
            )

        # If psi is too small, try to fix
        if psi < self.psi_grid[0]:
            if (
                np.argwhere(qpsi_grid > (qpsi_grid[0] + 1e-3)).squeeze()[0] > 1
            ):  # multiple almost-identical q values in a row
                lin_interp_q = np.polyfit(self.psi_grid[:30], qpsi_grid[:30], 1)
                psi_fixed = self.psi_grid[
                    np.argmin(np.abs(np.polyval(lin_interp_q, self.psi_grid[:30]) - q))
                ]
                if psi_fixed >= self.psi_grid[0]:
                    return psi_fixed

        raise ValueError(
            "Error: requested q=%1.3f is outside the gEQDSK range (q_min = %1.3f) and was unable to be fixed"
            % (q, qpsi_grid[0])
        )
