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


def detect_cocos(eqdsk: GEQDSKFile, sign_RphiZ: int | None = 1) -> int | None:
    """Detect the COCOS of a given GEQDSK file, or None if it cannot be determined.
    Cannot distinguish between odd and even COCOS since sigma_RphiZ is not recoverable from a geqdsk file.
    See `Sauter et al, 2013 <https://doi.org/10.1016/j.cpc.2012.09.010>`_.
    Also https://crppwww.epfl.ch/~sauter/cocos/ and https://crppwww.epfl.ch/~sauter/cocos/Sauter_COORD_CONVENTIONS_COCOS_2012_updated_after_reprint_for_Appendices_and_refs.pdf

    COCOS is defined by the following:
    - e_Bp: 0 if psi in Wb/rad, 1 if psi in Weber
    - sign_Bp: sign of poloidal magnetic field (Bp)
    - sign_RphiZ: +1 if (R, phi, Z), -1 if (R, Z, phi)
    - sign_rhotp: +1 if (rho, theta, phi), -1 if (rho, phi, theta)

    sign_rhotp uses Sauter Table I: sign(q) = sign_rhotp * sign(Ip * B0),
    so sign_rhotp = sign(q) * sign(Ip) * sign(B0)
    When q is stored as abs(q) (sign(q)=+1, as EFIT does) this reduces to the discharge helicity sign(Ip * B0)
    when q is signed it recovers sign_rhotp directly, which is what COCOS conversion encodes.

    NOTE: sigma_RphiZ cannot be determined from a geqdsk file alone. Both bcentr and
    fpol transform identically under a sigma_RphiZ change (Sauter Eq. 45), so their ratio
    is always +1 in a self-consistent file. This function assumes sigma_RphiZ=+1, the
    standard (R, phi, Z) convention used by geqdsk-producing codes (EFIT, LIUQE,
    CHEASE, etc.). It therefore only returns odd COCOS values (1,3,5,7,11,13,15,17).
    """
    # Initial checks to ensure the present timeslice has a decent equilibrium

    # If psi range is too small, pprime/ffprime profiles are flat
    # and the e_Bp check won't be consistent
    if np.abs(eqdsk.sibdry - eqdsk.simagx) < 0.01:
        logger.warning("sibdry and simagx are very close, unable to determine COCOS")
        return None

    # Startup slices can have sibdry outside the range of psi values,
    # these don't really make sense to run on since the equilibrium is not well-formed, so just skip them.
    psi_increasing = float(np.sign(eqdsk.sibdry - eqdsk.simagx))
    if psi_increasing > 0:
        if eqdsk.sibdry > np.max(eqdsk.psirz):
            logger.warning(
                "sibdry > max(psirz) with increasing psi from axis to boundary, unable to determine COCOS"
            )
            return None
        # simagx < min(psirz) is allowed
    else:
        if eqdsk.sibdry < np.min(eqdsk.psirz):
            logger.warning(
                "sibdry < min(psirz) with decreasing psi from axis to boundary, unable to determine COCOS"
            )
            return None
        # simagx > max(psirz) is allowed

    sign_Ip = np.sign(float(eqdsk.current))
    sign_B0 = np.sign(float(eqdsk.bcentr))
    # From table III: sign(dpsi) = sign_Bp * sign_Ip
    sign_Bp = int(psi_increasing * sign_Ip)

    def _e_Bp(eqdsk):
        # Detect e_Bp via the Grad-Shafranov residual.
        # The GS equation for psi in Wb/rad (e_Bp=0) is:
        #   Delta*(psi) = -(mu_0*R^2*pprime + ffprime)
        # If psi is in Weber (e_Bp=1) the same stored pprime/ffprime satisfy:
        #   Delta*(psi) = -(2*pi)^2 * (mu_0*R^2*pprime + ffprime)
        # So alpha = lhs / rhs_ebp0 is ~1 for e_Bp=0 and ~(2*pi)^2 for e_Bp=1.
        # Threshold at the geometric mean of 1 and (2*pi)^2, which is 2*pi.
        R_1d = np.asarray(eqdsk.r_grid[:, 0], dtype=float)
        Z_1d = np.asarray(eqdsk.z_grid[0, :], dtype=float)
        psirz = np.asarray(eqdsk.psirz, dtype=float)
        # freeqdsk stores psirz as (nR, nZ)
        # Only transpose a grid that is unambiguously (nZ, nR), square grids are already (nR, nZ).
        if psirz.shape == (len(Z_1d), len(R_1d)) and psirz.shape != (
            len(R_1d),
            len(Z_1d),
        ):
            psirz = psirz.T

        simagx = float(eqdsk.simagx)
        sibdry = float(eqdsk.sibdry)

        # LHS: Delta*(psi) = d2psi/dR2 - (1/R)*dpsi/dR + d2psi/dZ2, evaluated
        # analytically from a smooth bicubic spline (less noisy than finite differences).
        psirz_spline = RectBivariateSpline(R_1d, Z_1d, psirz, kx=3, ky=3, s=0)
        R_2d, Z_2d = np.meshgrid(R_1d, Z_1d, indexing="ij")
        lhs_gs = (
            psirz_spline.ev(R_2d, Z_2d, dx=2, dy=0)
            - psirz_spline.ev(R_2d, Z_2d, dx=1, dy=0) / R_2d
            + psirz_spline.ev(R_2d, Z_2d, dx=0, dy=2)
        )

        # RHS from the stored profiles, mapped over normalized psi (axis=0, boundary=1).
        # Clipping keeps the np.interp x-axis increasing regardless of psi sign convention.
        psi_norm_2d = (psirz - simagx) / (sibdry - simagx)
        psi_norm_1d = np.linspace(0.0, 1.0, len(eqdsk.pprime))
        pprime_2d = np.interp(
            np.clip(psi_norm_2d, 0.0, 1.0), psi_norm_1d, np.asarray(eqdsk.pprime, float)
        )
        ffprime_2d = np.interp(
            np.clip(psi_norm_2d, 0.0, 1.0),
            psi_norm_1d,
            np.asarray(eqdsk.ffprime, float),
        )
        rhs_gs = -(mu_0 * R_2d**2 * pprime_2d + ffprime_2d)

        # Use only the plasma core: away from the magnetic axis (small signal) and the
        # boundary/X-point (where the spline gradients and GS residual break down).
        mask = (psi_norm_2d >= 0.05) & (psi_norm_2d <= 0.95)
        rhs_max = np.max(np.abs(rhs_gs[mask])) if np.any(mask) else 0.0
        if rhs_max == 0:
            return 0
        mask &= np.abs(rhs_gs) > 0.05 * rhs_max

        # Least-squares slope of lhs vs rhs over the masked core (averages out grid noise).
        lhs_valid = lhs_gs[mask]
        rhs_valid = rhs_gs[mask]
        alpha = np.dot(lhs_valid, rhs_valid) / np.dot(rhs_valid, rhs_valid)
        logger.debug(f"Grad-Shafranov scaling factor alpha (core): {alpha:.4f}")

        return 0 if alpha < 2 * np.pi else 1

    e_Bp = _e_Bp(eqdsk)

    # From Sauter Table I: sign(q) = sign_rhotp * sign(Ip * B0), so
    # sign_rhotp = sign(q) * sign(Ip) * sign(B0)
    # With abs(q) (EFIT) sign(q)=+1 and this is just the helicity sign(Ip * B0)
    # with signed q it recovers sign_rhotp, which is what COCOS conversion encodes (conversion changes only sign(q), not Ip or B0).
    # When bcentr=0, infer sign_B0 from fpol. With sigma_RphiZ=+1 (geqdsk convention),
    # F = R*B_phi has the same sign as B0.
    if sign_B0 != 0:
        sign_B0_eff = int(sign_B0)
    else:
        sign_B0_eff = int(np.sign(np.nanmean(eqdsk.fpol)))
        logger.warning("bcentr=0, inferring sign_B0=%d from fpol" % sign_B0_eff)
    sign_q = int(np.sign(np.nanmean(np.asarray(eqdsk.qpsi, dtype=float)))) or 1
    sign_rhotp = sign_q * int(sign_Ip) * sign_B0_eff

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
        if cocos_input is None:
            raise ValueError(
                "Could not determine COCOS for the given GEQDSK."
                "Either specify a cocos_input or filter out timeslices where COCOS cannot be determined (e.g. startup slices)"
            )

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
    e_Bp_i, sign_Bp_i, sign_RphiZ_i, _sign_rhotp_i = cocos_params[cocos_input]
    e_Bp_o, sign_Bp_o, sign_RphiZ_o, sign_rhotp_o = cocos_params[cocos_target]

    # Conversion factors from Sauter 2013, Appendix C Eq. (45).
    # Pure coordinate conversion (SI to SI, no normalization change):
    #   sigma_tilde_Ip  = sigma_RphiZ_out * sigma_RphiZ_in   (Eq. 39)
    #   sigma_tilde_B0  = sigma_RphiZ_out * sigma_RphiZ_in   (Eq. 40)
    #   sigma_tilde_Bp  = sigma_Bp_out * sigma_Bp_in         (Eq. 41)
    #   e_tilde_Bp      = e_Bp_out - e_Bp_in                 (Eq. 41)
    sigma_tilde_Ip = sign_RphiZ_o * sign_RphiZ_i
    sigma_tilde_B0 = sign_RphiZ_o * sign_RphiZ_i
    sigma_tilde_Bp = sign_Bp_o * sign_Bp_i

    # psi_out = sigma_tilde_Ip * sigma_tilde_Bp * (2*pi)^(e_Bp_o - e_Bp_i) * psi_in
    psi_factor = sigma_tilde_Ip * sigma_tilde_Bp * (2 * np.pi) ** (e_Bp_o - e_Bp_i)

    # F_out = sigma_tilde_B0 * F_in (F = R*B_phi depends on phi direction)
    F_factor = sigma_tilde_B0

    # pprime = dp/dpsi and ffprime = F*dF/dpsi both scale as 1/psi_factor
    # (sigma_tilde_B0^2 = 1, so F sign cancels in ffprime)

    # I_out = sigma_tilde_Ip * I_in
    current = float(eqdsk.current) * sigma_tilde_Ip
    # B_out = sigma_tilde_B0 * B_in
    bcentr = float(eqdsk.bcentr) * sigma_tilde_B0

    # q: EFIT and many codes store abs(q), so the input sign is unreliable.
    # Compute the correct sign from Sauter Eq. (22):
    #   sign(q) = sign(Ip) * sign(B0) * sigma_rhotp
    # using the already-transformed Ip and B0 for the target COCOS.
    sign_q_target = int(np.sign(current) * np.sign(bcentr)) * sign_rhotp_o
    qpsi = np.abs(np.array(eqdsk.qpsi, dtype=float)) * sign_q_target

    new_eqdsk = GEQDSKFile(
        # Unchanged (geometry and pressure are coordinate-independent)
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
        pres=eqdsk.pres,
        nbdry=eqdsk.nbdry,
        nlim=eqdsk.nlim,
        rbdry=eqdsk.rbdry,
        zbdry=eqdsk.zbdry,
        rlim=eqdsk.rlim,
        zlim=eqdsk.zlim,
        # COCOS-dependent
        bcentr=bcentr,
        cpasma=current,
        simagx=float(eqdsk.simagx) * psi_factor,
        sibdry=float(eqdsk.sibdry) * psi_factor,
        psi=np.array(eqdsk.psirz, dtype=float) * psi_factor,
        fpol=np.array(eqdsk.fpol, dtype=float) * F_factor,
        ffprime=np.array(eqdsk.ffprime, dtype=float) / psi_factor,
        pprime=np.array(eqdsk.pprime, dtype=float) / psi_factor,
        qpsi=qpsi,
    )

    return new_eqdsk


class EquilibriumField:
    def __init__(self, eqdsk, cocos_input: int | None = None, lam=1e-7):
        eqdsk = convert_cocos(eqdsk, cocos_target=1, cocos_input=cocos_input)  # Convert to COCOS 1 internally

        self.eqdsk = eqdsk
        self.psi = RectBivariateSpline(
            eqdsk.r_grid[:, 0], eqdsk.z_grid[0, :], eqdsk.psi, kx=3, ky=3, s=0
        )

        # Linear grid of psi for 1D profiles
        # https://freeqdsk.readthedocs.io/en/stable/geqdsk.html
        self.psi_grid = np.linspace(eqdsk.simagx, eqdsk.sibdry, eqdsk.nx)

        def _smooth_qpsi_F(eqdsk, psi_grid, lam):
            # Switching to smoothing spline to avoid strange "discretization" jumps
            # Note: lam value (lower = less smoothing) should be reasonably consistent across q
            # profiles, but this is not certain. Lam=1e-7 works for q(psi) and F(psi) so far.

            # make_smoothing_spline requires strictly increasing x.
            # psi can decrease from axis to boundary (e.g. after COCOS conversion flips sign),
            # regardless of Ip sign. Check the actual direction and reverse if needed.
            if psi_grid[-1] < psi_grid[0]:
                psi_grid = psi_grid[::-1]
                qpsi = eqdsk.qpsi[::-1]
                fpol = eqdsk.fpol[::-1]
            else:
                qpsi = eqdsk.qpsi
                fpol = eqdsk.fpol

            # q(psi)
            qpsi_spline = make_smoothing_spline(psi_grid, qpsi, lam=lam, axis=0)
            # F(psi)
            F_spline = make_smoothing_spline(psi_grid, fpol, lam=lam, axis=0)

            return qpsi_spline, F_spline

        self.qpsi, self.F = _smooth_qpsi_F(eqdsk, self.psi_grid, lam)

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
        """
        is_increasing = (self.psi_grid[1] - self.psi_grid[0]) > 0

        if is_increasing:
            # positive-monotonic: index 0 at axis (low psi), index -1 at LCFS (high psi)

            # If psi is in range, return it unchanged
            if (psi >= self.psi_grid[0]) and (psi <= self.psi_grid[-1]):
                return psi

            # Decide by the requested q, not by where Newton landed: an out-of-range q can make
            # the solver overshoot to either psi bound, so a too-low q can land past the LCFS and
            # look like a q_max miss. Only raise q_max when q truly exceeds the profile maximum.
            if q > qpsi_grid[-1]:
                raise ValueError(
                    "Error: requested q=%1.3f is outside the gEQDSK range (q_max = %1.3f) and was unable to be fixed"
                    % (q, qpsi_grid[-1])
                )

            # q is below the on-axis value: try to fix when q is stepwise/grouped near the axis
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
        else:
            # negative-monotonic: index 0 at axis (high psi), index -1 at LCFS (low psi)

            # If psi is in range, return it unchanged
            if (psi >= self.psi_grid[-1]) and (psi <= self.psi_grid[0]):
                return psi

            # Decide by the requested q, not by where Newton landed (see increasing branch).
            if q > qpsi_grid[-1]:
                raise ValueError(
                    "Error: requested q=%1.3f is outside the gEQDSK range (q_max = %1.3f) and was unable to be fixed"
                    % (q, qpsi_grid[-1])
                )

            # q is below the on-axis value: try to fix when q is stepwise/grouped near the axis
            if (
                np.argwhere(qpsi_grid < (qpsi_grid[0] + 1e-3)).squeeze().size > 1
            ):  # multiple almost-identical q values in a row
                lin_interp_q = np.polyfit(self.psi_grid[:30], qpsi_grid[:30], 1)
                psi_fixed = self.psi_grid[
                    np.argmin(np.abs(np.polyval(lin_interp_q, self.psi_grid[:30]) - q))
                ]
                if psi_fixed <= self.psi_grid[0]:
                    return psi_fixed

            raise ValueError(
                "Error: requested q=%1.3f is outside the gEQDSK range (q_min = %1.3f) and was unable to be fixed"
                % (q, qpsi_grid[0])
            )
