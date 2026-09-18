import numpy as np
from freeqdsk.geqdsk import GEQDSKFile
from loguru import logger
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline, make_smoothing_spline
from scipy.optimize import newton, root_scalar

from synthwave.magnetic_geometry.utils import (
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
)


def biot_savart_cartesian(
    eval_point: np.ndarray, filament_points: np.ndarray, filament_current: float
):
    """Biot-savart law in cartesian coordinates.

    Segment-midpoint rule, one current element per polyline segment.
    """
    dl = filament_points[1:] - filament_points[:-1]
    midpoints = 0.5 * (filament_points[1:] + filament_points[:-1])
    r_prime = eval_point - midpoints
    r_prime_norm = np.linalg.norm(r_prime, axis=1)
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

    def _e_Bp_new(eqdsk):
        # Detect e_Bp via Grad-Shafranov residual.
        # The GS equation for psi in Wb/rad (e_Bp=0) is:
        #   Delta*(psi) = -(mu_0*R^2*pprime + ffprime)
        # If psi is in Weber (e_Bp=1) the same stored pprime/ffprime satisfy:
        #   Delta*(psi) = -(2*pi)^2 * (mu_0*R^2*pprime + ffprime)
        # Fit alpha such that lhs = alpha * rhs_ebp0: alpha~1 -> e_Bp=0, alpha~(2*pi)^2 -> e_Bp=1.

        # 1. Set up the exact coordinate vectors and grids
        R_1d = np.array(eqdsk.r_grid[:, 0], dtype=float)
        Z_1d = np.array(eqdsk.z_grid[0, :], dtype=float)

        # Build 2D meshgrids for direct spline evaluation
        R_2d, Z_2d = np.meshgrid(R_1d, Z_1d, indexing="ij")

        psi_2d = np.array(eqdsk.psi, dtype=float)
        if psi_2d.shape == (len(Z_1d), len(R_1d)):
            psi_2d = psi_2d.T  # Ensure shape is (nR, nZ) to match Spline indexing

        # Reconstruct the exact 2D spline of psi for analytic differentiation
        psi_spline = RectBivariateSpline(R_1d, Z_1d, psi_2d, kx=3, ky=3, s=0)

        # 2. Evaluate LHS of the Grad-Shafranov Equation analytically via Spline
        # Delta*(psi) = d2psi/dR2 - (1/R)*dpsi/dR + d2psi/dZ2
        d2psi_dR2 = psi_spline.ev(R_2d, Z_2d, dx=2, dy=0)
        dpsi_dR = psi_spline.ev(R_2d, Z_2d, dx=1, dy=0)
        d2psi_dZ2 = psi_spline.ev(R_2d, Z_2d, dx=0, dy=2)

        lhs_gs = d2psi_dR2 - (dpsi_dR / R_2d) + d2psi_dZ2

        # 3. Map 1D profiles over the raw, physical linear Psi grid
        pprime_raw = np.array(eqdsk.pprime, dtype=float)
        ffprime_raw = np.array(eqdsk.ffprime, dtype=float)

        simagx = float(eqdsk.simagx)
        sibdry = float(eqdsk.sibdry)

        psi_1d_mesh = np.linspace(simagx, sibdry, len(pprime_raw))

        # Interpolate 1D profiles directly to the 2D raw psi map
        pprime_2d = np.interp(psi_2d, psi_1d_mesh, pprime_raw)
        ffprime_2d = np.interp(psi_2d, psi_1d_mesh, ffprime_raw)

        # Calculate the RHS assuming e_Bp = 0
        rhs_gs = -(mu_0 * R_2d**2 * pprime_2d + ffprime_2d)

        # 4. Calculate Normalized Psi to isolate the core plasma region
        # Axis = 0.0, LCFS Boundary = 1.0
        psi_norm_2d = (psi_2d - simagx) / (sibdry - simagx)

        # Flatten arrays for regression
        lhs_flat = lhs_gs.ravel()
        rhs_flat = rhs_gs.ravel()
        psi_norm_flat = psi_norm_2d.ravel()

        rhs_max = np.max(np.abs(rhs_flat))
        if rhs_max == 0:
            return 0

        # Physical Mask: Only include points well inside the plasma core
        # We cut off at psi_norm = 0.95 to stay clear of the boundary pedestal
        # and X-point numerical artifacts where the spline gradients can get noisy.
        plasma_core_mask = (psi_norm_flat >= 0.0) & (psi_norm_flat <= 0.95)

        # Signal Noise Mask: Ignore regions where RHS is fundamentally zero
        signal_mask = np.abs(rhs_flat) > 0.05 * rhs_max

        # Combined clean mask
        final_mask = plasma_core_mask & signal_mask

        lhs_valid = lhs_flat[final_mask]
        rhs_valid = rhs_flat[final_mask]

        # Global linear regression: alpha = sum(LHS * RHS) / sum(RHS^2)
        alpha = np.dot(lhs_valid, rhs_valid) / np.dot(rhs_valid, rhs_valid)

        logger.debug(
            f"Detected Grad-Shafranov scaling alpha factor (Core Plasma Only): {alpha:.4f}"
        )

        # Distinguish e_Bp = 0 (alpha ~ 1) from e_Bp = 1 (alpha ~ 39.4) using geometric mean (2*pi)
        e_Bp = 0 if alpha < 2 * np.pi else 1

        return e_Bp

    # e_Bp = _e_Bp(eqdsk)
    e_Bp = _e_Bp_new(eqdsk)

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
    # If bcentr is zero, infer the sign from fpol (which is always nonzero in a valid equilibrium).
    if bcentr == 0:
        sign_bcentr = int(np.sign(np.nanmean(eqdsk.fpol))) * sigma_tilde_B0
    else:
        sign_bcentr = int(np.sign(bcentr))
    sign_q_target = int(np.sign(current)) * sign_bcentr * sign_rhotp_o
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
        eqdsk = convert_cocos(
            eqdsk, cocos_target=1, cocos_input=cocos_input
        )  # Convert to COCOS 1 internally

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
            # |q(psi)|
            qpsi_abs_spline = make_smoothing_spline(
                psi_grid, np.abs(qpsi), lam=lam, axis=0
            )
            # F(psi)
            F_spline = make_smoothing_spline(psi_grid, fpol, lam=lam, axis=0)

            return qpsi_spline, qpsi_abs_spline, F_spline

        self.qpsi, self.qpsi_abs, self.F = _smooth_qpsi_F(eqdsk, self.psi_grid, lam)

    def get_field_at_point(self, R, Z) -> np.ndarray:
        # Bp = Br + Bz = (d(psi)/dZ - d(psi)/dR) / R
        psir = self.psi.ev(R, Z, dx=1, dy=0)
        psiz = self.psi.ev(R, Z, dx=0, dy=1)

        Br = psiz / R
        Bz = -psir / R
        Bt = self.F(self.psi.ev(R, Z)) / R

        return np.array([Br, Bt, Bz])

    def get_psi_of_q_old(self, q):
        """Get psi corresponding to a given q. Works in |q| space."""
        q_abs = np.abs(q)
        qpsi_grid = self.qpsi_abs(self.psi_grid)
        psi_guess = self.psi_grid[np.argmin(np.abs(qpsi_grid - q_abs))]
        psi = newton(
            func=lambda psi: self.qpsi_abs(psi) - q_abs,
            x0=psi_guess,
            fprime=lambda psi: self.qpsi_abs.derivative(1)(psi),
            maxiter=400,
            tol=1e-3,
        )

        # psi
        psi = self.core_psi_consistency_check(qpsi_grid, psi, q_abs)

        return psi

    def get_psi_of_q_raw(self, q):
        # Simple interpolation of raw |q|-psi grid to get an initial guess for psi(q)
        # For "regular" q-profiles, this is usually sufficient.
        q_abs = np.abs(q)
        qpsi_raw = np.abs(np.array(self.eqdsk.qpsi, dtype=float))
        psi_raw = np.array(self.psi_grid, dtype=float)
        if not (np.all(np.diff(qpsi_raw) >= 0) or np.all(np.diff(qpsi_raw) <= 0)):
            raise ValueError(
                "Raw abs(qpsi) profile is not monotonic and cannot be inverted safely"
            )

        if q_abs < qpsi_raw.min() or q_abs > qpsi_raw.max():
            raise ValueError(
                "Requested abs(q) is outside the raw abs(qpsi) range: %s not in [%s, %s]"
                % (q_abs, qpsi_raw.min(), qpsi_raw.max())
            )

        return float(np.interp(q_abs, qpsi_raw, psi_raw))

    def get_psi_of_q(self, q):
        """
        Get psi corresponding to a given |q|
        Slightly improved to try and use a bounded solver if possible,
        otherwise fall back to unbounded Newton's method
        The advantages of this is that it can circumvent some unusual, high m/n
        cases where the solver can get stuck near x-points, and non-monotonic
        q-profiles (based on DIII-D tests)

        """
        q_abs = np.abs(q)

        # Make an initial guess based on the smoothed |q|-psi grid.
        qpsi_grid = self.qpsi_abs(self.psi_grid)
        psi_guess_index = np.argmin(np.abs(qpsi_grid - q_abs))
        psi_guess = self.psi_grid[psi_guess_index]

        q_at_guess = float(self.qpsi_abs(psi_guess))
        if np.isclose(q_at_guess, q_abs, atol=1e-12):
            psi = psi_guess
        else:
            # Check if initial guess is within bounds of Psi
            if psi_guess_index == 0 or psi_guess_index == len(self.psi_grid) - 1:
                raise ValueError(
                    "Initial guess for psi is out of bounds. Requested abs(q)=%1.3f is outside the gEQDSK range (q_min = %1.3f, q_max = %1.3f)."
                    % (q_abs, qpsi_grid.min(), qpsi_grid.max())
                )

            # Bound possible psi values around initial guess from psi grid
            psi_lo = (
                self.psi_grid[psi_guess_index - 5]
                if psi_guess_index - 5 >= 0
                else self.psi_grid[0]
            )
            psi_hi = (
                self.psi_grid[psi_guess_index + 10]
                if psi_guess_index + 10 < len(self.psi_grid)
                else self.psi_grid[-1]
            )

            a = min(psi_lo, psi_hi)
            b = max(psi_lo, psi_hi)

            def fn_psi(psi):
                return self.qpsi_abs(psi) - q_abs

            # Use a bounded solver if possible, otherwise fall back to unbounded Newton's method
            bracket_found = np.sign(fn_psi(a)) != np.sign(fn_psi(b))
            if bracket_found:
                result = root_scalar(
                    fn_psi,
                    bracket=[a, b],
                    method="toms748",
                    xtol=1e-10,
                    maxiter=200,
                )
                psi = float(result.root)
            else:
                psi = newton(
                    func=lambda psi: np.abs(self.qpsi_abs(psi) - q_abs),
                    x0=psi_guess,
                    fprime=lambda psi: self.qpsi_abs.derivative(1)(psi),
                    maxiter=800,
                    tol=1e-10,
                )

        # Ensure that the final psi value is consistent with the |qpsi| grid
        psi = self.core_psi_consistency_check(qpsi_grid, psi, q_abs)
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

            q_max = np.max(qpsi_grid)

            # Past LCFS: only raise q_max when q truly exceeds the profile maximum.
            if psi > self.psi_grid[-1] and q > q_max:
                raise ValueError(
                    "Error: requested q=%1.3f is outside the gEQDSK range (q_max = %1.3f) and was unable to be fixed"
                    % (q, q_max)
                )

            # Past axis (or LCFS with q in range): try to fix when q is stepwise/grouped near the axis
            if (
                np.argwhere(qpsi_grid < (qpsi_grid[0] + 1e-3)).squeeze().size > 1
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
            q_max = np.max(qpsi_grid)
            if psi < self.psi_grid[-1] and q > q_max:
                raise ValueError(
                    "Error: requested q=%1.3f is outside the gEQDSK range (q_max = %1.3f) and was unable to be fixed"
                    % (q, q_max)
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
