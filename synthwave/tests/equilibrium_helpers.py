"""Shared helpers for tests and benchmarks that need real equilibria."""

from fractions import Fraction

import numpy as np
from freeqdsk.geqdsk import GEQDSKFile
from scipy.optimize import newton

from synthwave.magnetic_geometry.equilibrium_field import EquilibriumField


def tars_slice_to_eqdsk(
    ds_eq, shot: int = 82878, comment: str = "TCV LIUQE"
) -> GEQDSKFile:
    """GEQDSKFile from a single time slice of a TARS input dataset.

    The 1D profiles are resampled to nx points as the gEQDSK format requires, with
    non-finite entries filled by linear interpolation from their valid neighbours.
    """
    nx = len(ds_eq["r_grid"])
    ny = len(ds_eq["z_grid"])

    def _resample_to_nx(arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr, dtype=float)
        valid = np.isfinite(arr)
        if not np.any(valid):
            return np.zeros(nx)
        if not np.all(valid):
            xs = np.arange(len(arr))
            arr = np.interp(xs, xs[valid], arr[valid])
        if len(arr) == nx:
            return arr
        psi_old = np.linspace(0, 1, len(arr))
        psi_new = np.linspace(0, 1, nx)
        return np.interp(psi_new, psi_old, arr)

    return GEQDSKFile(
        comment=comment,
        shot=shot,
        nx=nx,
        ny=ny,
        rdim=float(ds_eq["r_grid"][-1] - ds_eq["r_grid"][0]),
        zdim=float(ds_eq["z_grid"][-1] - ds_eq["z_grid"][0]),
        rcentr=float(ds_eq["r_grid"][nx // 2]),
        rleft=float(ds_eq["r_grid"][0]),
        zmid=float(ds_eq["z_grid"][ny // 2]),
        rmagx=float(ds_eq["rmagx"]),
        zmagx=float(ds_eq["zmagx"]),
        simagx=float(ds_eq["simagx"]),
        sibdry=float(ds_eq["sibdry"]),
        bcentr=float(ds_eq["bcentr"]),
        cpasma=float(ds_eq["current"]),
        fpol=_resample_to_nx(ds_eq["fpol"].values),
        pres=_resample_to_nx(ds_eq["pres"].values),
        ffprime=_resample_to_nx(ds_eq["ffprime"].values),
        pprime=_resample_to_nx(ds_eq["pprime"].values),
        psi=ds_eq["psirz"].values,
        qpsi=_resample_to_nx(ds_eq["qpsi"].values),
        nbdry=len(ds_eq["rbdry"]),
        nlim=0,
        rbdry=ds_eq["rbdry"].values,
        zbdry=ds_eq["zbdry"].values,
        rlim=[],
        zlim=[],
    )


def reference_trace_scalar_newton(
    eq_field: EquilibriumField,
    mode: tuple[int, int],
    num_points: int,
    helicity_sign: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """EquilibriumFilamentTracer.trace (TraceType.AVERAGE) before the vectorized rewrite.

    One scipy newton per poloidal point, marching from the previous point, tol=1e-3 m,
    then trapezoid phi integration and the rescale to 2 pi m / n.

    Returns:
        filament_points (num_points, 3) as (R, phi, Z), filament_etas, and the integrated
        phi at the last point before the rescale (the closure diagnostic).
    """
    m, n = mode
    if helicity_sign is None:
        helicity_sign = 1 if n >= 0 else -1

    ratio = Fraction(m, n)
    m_local = np.abs(ratio.numerator)
    n_local = ratio.denominator
    psi_q = eq_field.get_psi_of_q(np.abs(m_local / n_local))

    sign_Ip = int(np.sign(float(eq_field.eqdsk.cpasma)))
    sign_Bt = int(np.sign(float(eq_field.F(psi_q))))

    filament_etas = np.linspace(0, 2 * np.pi, num_points)
    poloidal_points = np.zeros((num_points, 3))

    Z_start = eq_field.eqdsk.zmagx
    R_guess = eq_field.eqdsk.rmagx + 0.1
    R_start = newton(
        func=lambda R: eq_field.psi.ev(R, Z_start) - psi_q,
        x0=R_guess,
        fprime=lambda R: eq_field.psi.ev(R, Z_start, dx=1, dy=0),
        maxiter=800,
        tol=1e-3,
    )

    def _R_a(eta, a):
        return eq_field.eqdsk.rmagx + (a * np.cos(eta))

    def _Z_a(eta, a):
        return eq_field.eqdsk.zmagx + (-sign_Ip * helicity_sign) * (a * np.sin(eta))

    def psi_prime_a(eta, a):
        R = _R_a(eta, a)
        Z = _Z_a(eta, a)
        return eq_field.psi.ev(R, Z, dx=1, dy=0) * np.cos(eta) + (
            -sign_Ip * helicity_sign
        ) * eq_field.psi.ev(R, Z, dx=0, dy=1) * np.sin(eta)

    for i, eta in enumerate(filament_etas):
        if i == 0:
            R_prev = R_start
            Z_prev = Z_start
        else:
            R_prev = poloidal_points[i - 1, 0]
            Z_prev = poloidal_points[i - 1, 1]
        a_guess = np.sqrt(
            (R_prev - eq_field.eqdsk.rmagx) ** 2 + (Z_prev - eq_field.eqdsk.zmagx) ** 2
        )
        a_next = newton(
            func=lambda a: eq_field.psi.ev(_R_a(eta, a), _Z_a(eta, a)) - psi_q,
            x0=a_guess,
            fprime=lambda a: psi_prime_a(eta, a),
            maxiter=800,
            tol=1e-3,
        )
        poloidal_points[i, :] = [_R_a(eta, a_next), _Z_a(eta, a_next), a_next]

    R = poloidal_points[:, 0]
    Z = poloidal_points[:, 1]
    B = eq_field.get_field_at_point(R, Z)

    dl = np.sqrt(np.diff(R) ** 2 + np.diff(Z) ** 2)
    dphi_dl = B[1] / (R * np.sqrt(B[0] ** 2 + B[2] ** 2))
    d_phi = helicity_sign * dl * 0.5 * (dphi_dl[:-1] + dphi_dl[1:])
    phi = np.concatenate([[0.0], np.cumsum(d_phi)])

    known_phi_end = helicity_sign * sign_Bt * 2 * np.pi * m_local / n_local
    phi_end_raw = float(phi[-1])
    correction_factor = known_phi_end / (phi[-1] - phi[0])
    phi = (phi - phi[0]) * correction_factor

    return np.column_stack((R, phi, Z)), filament_etas, phi_end_raw
