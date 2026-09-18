from abc import ABC, abstractmethod
from enum import Enum
from fractions import Fraction
from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger
from scipy.interpolate import make_interp_spline
from sympy import nextprime

from synthwave.magnetic_geometry.equilibrium_field import (
    EquilibriumField,
)
from synthwave.magnetic_geometry.utils import cylindrical_to_cartesian


def filament_offsets(num_filaments: int) -> np.ndarray:
    """Toroidal starting angle [rad] of each filament copy, evenly spaced over one turn."""
    return np.linspace(0, 2 * np.pi, num_filaments, endpoint=False)


def filament_currents(mode: tuple[int, int], num_filaments: int) -> np.ndarray:
    """Complex current of each toroidally offset filament copy for a rotating (m, n) wave.

    I_k = exp(i * n * phi_k) with phi_k from filament_offsets. The sign of n sets the
    toroidal rotation direction, which is what a toroidal sensor array measures.
    The poloidal direction follows from the field helicity and is set by the trace,
    so m carries no sign.

    The winding uses the UNREDUCED n:
    geometry reduces (m, n) to lowest terms (same rational surface and field lines),
    but a non-coprime mode (k*m, k*n) is the k-th harmonic of the (m, n) mode.
    Same filaments, current pattern winding k times faster. For coprime modes nothing changes.

    Modes on one rational surface therefore share a filament trace and a unit-current
    flux matrix and differ only in this current vector.
    """
    return np.exp(1j * mode[1] * filament_offsets(num_filaments))


class FilamentTraceError(RuntimeError):
    """The rational surface could not be traced as a closed field line."""


def ray_lengths_to_grid_edge(R0, Z0, cos_eta, sin_eta, R_grid, Z_grid):
    """Distance from (R0, Z0) along each ray direction to the first edge of the R-Z grid."""
    with np.errstate(divide="ignore"):
        t_R = np.where(
            cos_eta > 0,
            (R_grid[-1] - R0) / cos_eta,
            np.where(cos_eta < 0, (R_grid[0] - R0) / cos_eta, np.inf),
        )
        t_Z = np.where(
            sin_eta > 0,
            (Z_grid[-1] - Z0) / sin_eta,
            np.where(sin_eta < 0, (Z_grid[0] - Z0) / sin_eta, np.inf),
        )
    return np.minimum(t_R, t_Z) * (1 - 1e-9)


def ray_lengths_to_boundary(R0, Z0, cos_eta, sin_eta, rbdry, zbdry):
    """Distance from (R0, Z0) along each ray direction to the plasma boundary outline.

    The outline radius is interpolated linearly in the poloidal angle about (R0, Z0),
    which assumes the boundary is smooth-ish about the axis. Returns inf everywhere when
    fewer than three finite boundary points are available.
    """
    rbdry = np.asarray(rbdry, dtype=float)
    zbdry = np.asarray(zbdry, dtype=float)
    finite = np.isfinite(rbdry) & np.isfinite(zbdry)
    rbdry, zbdry = rbdry[finite], zbdry[finite]
    if len(rbdry) < 3:
        return np.full(len(cos_eta), np.inf)
    theta_bdry = np.arctan2(zbdry - Z0, rbdry - R0)
    radius_bdry = np.hypot(rbdry - R0, zbdry - Z0)
    order = np.argsort(theta_bdry)
    theta_bdry, radius_bdry = theta_bdry[order], radius_bdry[order]
    theta_periodic = np.concatenate(
        [theta_bdry - 2 * np.pi, theta_bdry, theta_bdry + 2 * np.pi]
    )
    return np.interp(
        np.arctan2(sin_eta, cos_eta), theta_periodic, np.tile(radius_bdry, 3)
    )


def rational_surface_radii(
    eq_field: EquilibriumField,
    psi_q: float,
    cos_eta: np.ndarray,
    sin_eta: np.ndarray,
    num_radial_samples: int = 16,
    psi_rtol: float = 1e-8,
    max_iter: int = 20,
) -> np.ndarray:
    """Minor radius a(eta) where psi = psi_q along rays from the magnetic axis.

    Every ray (cos_eta, sin_eta) in the R-Z plane is sampled from the axis to 5 percent
    past the boundary outline (or the grid edge). Inside the boundary psi is monotonic
    along a ray, so the first sign change of psi - psi_q brackets the one crossing.
    The bracket is refined with a batched Newton iteration on the psi spline,
    and the root must stay inside its bracket cell.

    Raises:
        FilamentTraceError: a ray has no crossing inside the boundary (the surface is not
            closed inside the plasma), Newton does not converge, or a root leaves its cell.
    """
    eqdsk = eq_field.eqdsk
    R0, Z0 = eqdsk.rmagx, eqdsk.zmagx
    delta_psi = abs(eqdsk.sibdry - eqdsk.simagx)

    a_edge = ray_lengths_to_grid_edge(
        R0, Z0, cos_eta, sin_eta, eqdsk.r_grid[:, 0], eqdsk.z_grid[0, :]
    )
    a_lcfs = ray_lengths_to_boundary(R0, Z0, cos_eta, sin_eta, eqdsk.rbdry, eqdsk.zbdry)
    a_max = np.minimum(1.05 * a_lcfs, a_edge)

    # Bracket: first sign change of psi - psi_q outward from the axis
    a_samples = a_max[:, None] * np.linspace(0, 1, num_radial_samples)[None, :]
    f_samples = (
        eq_field.psi.ev(
            R0 + a_samples * cos_eta[:, None], Z0 + a_samples * sin_eta[:, None]
        )
        - psi_q
    )
    sign_change = f_samples[:, 1:] * f_samples[:, :1] <= 0
    has_crossing = sign_change.any(axis=1)
    if not has_crossing.all():
        raise FilamentTraceError(
            f"psi_q={psi_q:.6f} surface is not closed inside the boundary, "
            f"no crossing on {int(np.sum(~has_crossing))} of {len(cos_eta)} rays"
        )
    rays = np.arange(len(cos_eta))
    j_hi = np.argmax(sign_change, axis=1) + 1
    a_lo, a_hi = a_samples[rays, j_hi - 1], a_samples[rays, j_hi]
    f_lo, f_hi = f_samples[rays, j_hi - 1], f_samples[rays, j_hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = a_lo + (a_hi - a_lo) * f_lo / (f_lo - f_hi)
    a = np.where(np.isfinite(a), a, 0.5 * (a_lo + a_hi))

    # Batched Newton on psi(a) - psi_q with the analytic derivative along the ray
    for _ in range(max_iter):
        R = R0 + a * cos_eta
        Z = Z0 + a * sin_eta
        f = eq_field.psi.ev(R, Z) - psi_q
        residual = np.max(np.abs(f)) / delta_psi
        if residual < psi_rtol:
            break
        df = (
            eq_field.psi.ev(R, Z, dx=1, dy=0) * cos_eta
            + eq_field.psi.ev(R, Z, dx=0, dy=1) * sin_eta
        )
        a = a - f / df
    else:
        raise FilamentTraceError(
            f"psi_q={psi_q:.6f} Newton did not converge in {max_iter} iterations, "
            f"relative residual {residual:.2e}"
        )
    if not np.all(np.isfinite(a)) or np.any(a < a_lo) or np.any(a > a_hi):
        raise FilamentTraceError(
            f"psi_q={psi_q:.6f} Newton left the bracket of the first crossing on "
            f"{int(np.sum(~((a >= a_lo) & (a <= a_hi))))} rays"
        )
    return a


class FilamentTracer(ABC):
    """Abstract class for filament representation.

    Mode convention: The sign of n is the toroidal rotation direction, which a toroidal
    sensor array measures directly. m is the poloidal mode number and is never negative.
    The poloidal direction follows from the field helicity, so negative n traces the same
    field line antiparallel and winds its currents the other way (see filament_currents).
    """

    def __init__(
        self,
        mode: tuple[int, int],
        base_num_points: Optional[int] = 800,
        scale_points: Optional[bool] = True,
        prevent_synthetic_structure: Optional[bool] = True,
    ):
        self.m = mode[0]
        self.n = mode[1]

        num_points = base_num_points
        if scale_points:
            num_points = int(base_num_points * abs(self.m) / abs(self.n))
        if prevent_synthetic_structure:
            num_points = nextprime(num_points)

        self.num_points = num_points

    @abstractmethod
    def trace(self, num_points: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Trace the filament and return the points in cylindrical coordinates (R, phi, Z), and the corresponding eta values."""

    def get_filament_ds(
        self,
        num_filaments: int,
        coordinate_system: Optional[str] = "cartesian",
    ) -> xr.Dataset:
        """Generate points and corresponding currents for multiple filaments.

        Args:
            num_filaments (int): How many individual filaments to create
            coordinate_system (Optional[str], default = "cartesian"): Coordinate system for output points. Options are "cylindrical", "cartesian", or "toroidal".

        Returns:
            xr.Dataset: Dataset containing filament points and currents. Dimensions are 'filament' and 'point', with variables 'R', 'phi', 'Z' or 'x', 'y', 'z' or 'eta', 'phi', and 'current'.
        """

        if num_filaments <= 0:
            raise ValueError("num_filaments must be a positive integer")

        if coordinate_system not in ["cylindrical", "cartesian", "toroidal"]:
            raise ValueError(
                "coordinate_system must be either 'cylindrical', 'cartesian', or 'toroidal'"
            )

        # Start with a filament that has zero toroidal offset
        base_filament_points, filament_etas = self.trace()

        # Create toroidal offsets and corresponding currents
        starting_angles = filament_offsets(num_filaments)
        currents = filament_currents((self.m, self.n), num_filaments)

        all_filament_points = np.repeat(
            base_filament_points[np.newaxis, :, :], num_filaments, axis=0
        )  # Shape (num_filaments, N, 3)
        all_filament_points[:, :, 1] += starting_angles[
            :, np.newaxis
        ]  # Apply toroidal offsets

        if coordinate_system == "cylindrical":
            ds = xr.Dataset(
                data_vars={
                    "R": (("filament", "point"), all_filament_points[:, :, 0]),
                    "phi": (("filament", "point"), all_filament_points[:, :, 1]),
                    "Z": (("filament", "point"), all_filament_points[:, :, 2]),
                    "current": (("filament"), currents),
                },
                coords={
                    "filament": np.arange(num_filaments),
                    "point": np.arange(len(base_filament_points)),
                },
            )
        elif coordinate_system == "cartesian":
            cartesian_points = cylindrical_to_cartesian(
                all_filament_points[:, :, 0],
                all_filament_points[:, :, 1],
                all_filament_points[:, :, 2],
            )  # Shape (3, num_filaments, N)
            ds = xr.Dataset(
                data_vars={
                    "x": (("filament", "point"), cartesian_points[0, :, :]),
                    "y": (("filament", "point"), cartesian_points[1, :, :]),
                    "z": (("filament", "point"), cartesian_points[2, :, :]),
                    "current": (("filament"), currents),
                },
                coords={
                    "filament": np.arange(num_filaments),
                    "point": np.arange(len(base_filament_points)),
                },
            )
        elif coordinate_system == "toroidal":
            ds = xr.Dataset(
                data_vars={
                    "eta": (("point"), filament_etas),
                    "phi": (("filament", "point"), all_filament_points[:, :, 1]),
                    "current": (("filament"), currents),
                },
                coords={
                    "filament": np.arange(num_filaments),
                    "point": np.arange(len(filament_etas)),
                },
            )

        return ds

    def get_filament_list(
        self,
        num_filaments: int,
        coordinate_system: str = "cartesian",
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate a list of filaments, each represented as an array of shape (N, 3) in cylindrical coordinates.

        Args:
            num_filaments (int): Number of filaments to generate.
            coordinate_system (str, default = "cartesian"): Coordinate system for output points. Options are "cylindrical" or "cartesian".

        Returns:
            list[np.ndarray]: List of filaments, each of shape (N, 3) with columns [x, y, z] or [R, phi, Z].
            list[float]: List of filament currents.
        """
        if num_filaments <= 0:
            raise ValueError("num_filaments must be a positive integer")

        if coordinate_system not in ["cylindrical", "cartesian"]:
            raise ValueError(
                "coordinate_system must be either 'cylindrical' or 'cartesian'"
            )

        filament_points_ds = self.get_filament_ds(num_filaments, coordinate_system)

        names = (
            ("R", "phi", "Z") if coordinate_system == "cylindrical" else ("x", "y", "z")
        )
        points = np.stack(
            [filament_points_ds[name].values for name in names], axis=-1
        )  # Shape (num_filaments, N, 3)
        filament_list = list(points)
        current_list = filament_points_ds["current"].values.tolist()

        return filament_list, current_list

    def make_filament_spline(self):
        """Create a spline which puts eta in terms of phi for this filament."""
        filament_points, filament_etas = self.trace()

        spline = make_interp_spline(filament_points[:, 1], filament_etas)
        return spline


class ToroidalFilamentTracer(FilamentTracer):
    """Filament for toroidal approximation of the magnetic geometry."""

    def __init__(
        self,
        mode: tuple[int, int],
        R0: float,
        Z0: float,
        a: float,
        base_num_points: Optional[int] = 1000,
        scale_points: Optional[bool] = True,
        prevent_synthetic_structure: Optional[bool] = True,
        sign_Ip: Optional[int] = 1,
        sign_B0: Optional[int] = 1,
    ):
        """Initialize a toroidal filament with a circular cross-section.
        Follows COCOS 1 convention for tracing.

        Parameters
        ----------
        mode : tuple[int, int]
            Mode number (m, n)
        R0 : float
            Major radius of the magnetic axis
        Z0 : float
            Vertical position of the magnetic axis
        a : float
            Minor radius of the circular cross-section
        base_num_points : int, optional
            Base number of points to trace around the filament
        scale_points : bool, optional
            Whether to scale the number of points based on m/n ratio. If true, multiplies base_num_points by m/n to ensure adequate resolution.
        prevent_synthetic_structure : bool, optional
            Whether to adjust the number of points to the next prime number to avoid synthetic structures in simulations.
        sign_Ip : int, optional
            Sign of the plasma current. Default is +1.
        sign_B0 : int, optional
            Sign of the toroidal magnetic field. Default is +1.
        """
        super().__init__(
            mode, int(base_num_points), scale_points, prevent_synthetic_structure
        )
        self.R0 = R0
        self.Z0 = Z0
        self.a = a
        self.sign_Ip = sign_Ip
        self.sign_B0 = sign_B0

    def trace(self, num_points: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Create a circular filament in a toroidal geometry.
        This uses COCOS 1, where phi is CCW angle when viewed from the top and eta is CW when viewing the right poloidal cross section.

        Args:
            num_points (Optional[int]): Number of points to use for tracing a single poloidal turn

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing arrays describing the filament coordinates and corresponding eta values
        """
        if num_points is None:
            num_points = self.num_points
        mode_ratio = np.abs(self.m / self.n)
        phi = np.linspace(0, 2 * np.pi * mode_ratio, num_points)
        filament_etas = np.linspace(0, 2 * np.pi, num_points)
        R = self.R0 + (self.a * np.cos(filament_etas)) * self.sign_B0
        Z = self.Z0 - (self.a * np.sin(filament_etas)) * self.sign_Ip

        filament_points = np.column_stack((R, phi, Z))

        if self.n < 0:
            # trace should go antiparallel to the field
            # Flip the arrays and reverse the direction of filament etas
            filament_points = filament_points[::-1]
            filament_etas = -filament_etas

        return filament_points, filament_etas


class EquilibriumFilamentTracer(FilamentTracer):
    """Filament traced along an equilibrium magnetic field."""

    class TraceType(Enum):
        CYLINDRICAL = 0  # Cylindrical approximation of the magnetic geometry
        NAIVE = 1  # Naive tracing, following the rational surface but not the field
        AVERAGE = 2  # Field-line tracing, d(phi)/dl from the magnetic field integrated per segment (trapezoid)

    def __init__(
        self,
        mode: tuple[int, int],
        eq_field: EquilibriumField,
        base_num_points: Optional[int] = 601,
        scale_points: Optional[bool] = True,
        prevent_synthetic_structure: Optional[bool] = True,
        default_trace_type: TraceType = TraceType.AVERAGE,
        helicity_sign: Optional[int] = None,
        closure_rtol: float = 0.1,
    ):
        """Initialize an equilibrium filament.

        Parameters
        ----------
        mode : tuple[int, int]
            Mode number (m, n)
        eq_field : EquilibriumField
            EquilibriumField object containing the magnetic field data
        base_num_points : int, optional
            Number of points to trace around the filament
        scale_points : bool, optional
            Whether to scale the number of points based on m/n ratio. If true, multiplies base_num_points by m/n to ensure adequate resolution.
        prevent_synthetic_structure : bool, optional
            Whether to adjust the number of points to the next prime number to avoid synthetic structures in simulations.
        default_trace_type : TraceType, optional
            Default tracing method to use
        helicity_sign : int, optional
            Sign of the helicity used when following the field lines. A value of
            ``+1`` traces in the default direction, while ``-1`` reverses the direction.
            If ``None`` (default), the sign is inferred from the sign of ``n``: positive n
            traces parallel to the field, negative n antiparallel (the mirror mode rotating
            the other way toroidally). The sign of m is ignored.
        closure_rtol : float, optional
            Largest allowed relative miss of the integrated toroidal angle after one
            poloidal turn against 2 pi m / n. The miss equals the relative difference
            between the field line q of the traced surface and m / n.
            A trace beyond it raises FilamentTraceError instead of being rescaled.

        """
        super().__init__(
            mode,
            int(base_num_points),
            scale_points,
            prevent_synthetic_structure,
        )
        self.eq_field = eq_field
        self.default_trace_type = default_trace_type
        self.helicity_sign = (
            helicity_sign if helicity_sign is not None else (1 if self.n >= 0 else -1)
        )
        self.closure_rtol = closure_rtol
        self.trace_cache = {}

    def trace(
        self,
        num_points: Optional[int] = None,
        trace_type: TraceType = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace a filament along the equilibrium magnetic field.

        Parameters
        ----------
        num_points : int, optional
            Number of points to use for tracing a single poloidal turn of the filament.
            If ``None``, the value stored in ``self.num_points`` is used.
        trace_type : EquilibriumFilamentTracer.TraceType, optional
            Tracing strategy to use. This controls how the field is followed when
            computing the filament shape (cylindrical approximation, naive rational
            surface, or field-line tracing, see TraceType).

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing arrays describing the traced filament coordinates.

        """

        if num_points is None:
            num_points = self.num_points
        if trace_type is None:
            trace_type = self.default_trace_type

        key = (num_points, trace_type)
        if key in self.trace_cache:
            logger.info(
                f"Using cached filament trace for num_points={num_points} and trace_type={trace_type}"
            )
            return self.trace_cache[key]

        # Correction for m/n as integer multiples (otherwise leads to ``wandering'' filaments)
        ratio = Fraction(self.m, self.n)
        m_local = np.abs(ratio.numerator)
        n_local = ratio.denominator
        psi_q = self.eq_field.get_psi_of_q(np.abs(m_local / n_local))

        sign_Ip = int(np.sign(float(self.eq_field.eqdsk.cpasma)))
        sign_Bt = int(np.sign(float(self.eq_field.F(psi_q))))

        # Rays from the magnetic axis.
        # Z direction: Bz at the outboard midplane in COCOS 1 is -sign_Ip * |Bp|,
        # so field-parallel tracing (helicity_sign=+1) follows sign(Bz) = -sign_Ip and antiparallel tracing reverses it.
        sign_Z = -sign_Ip * self.helicity_sign
        filament_etas = np.linspace(0, 2 * np.pi, num_points)
        cos_eta = np.cos(filament_etas)
        sin_eta = sign_Z * np.sin(filament_etas)
        try:
            minor_radius = rational_surface_radii(
                self.eq_field, psi_q, cos_eta, sin_eta
            )
        except FilamentTraceError as err:
            raise FilamentTraceError(f"m={self.m} n={self.n}: {err}") from None
        poloidal_points = np.column_stack(
            (
                self.eq_field.eqdsk.rmagx + minor_radius * cos_eta,
                self.eq_field.eqdsk.zmagx + minor_radius * sin_eta,
                minor_radius,
            )
        )  # R, Z, a

        # alternative form removing the assumption that dl = r d_eta (that assumption holds only for circular cross-sections)
        def _d_phi_dl(dl, R, Bp, Bt):
            # https://youjunhu.github.io/research_notes/tokamak_equilibrium_htlatex/tokamak_equilibrium.html
            return (Bt * dl) / (R * Bp)

        # Finalize filament trace based on trace_type
        if trace_type == EquilibriumFilamentTracer.TraceType.CYLINDRICAL:
            # Circular cross section around the magnetic axis
            avg_minor_radius = np.mean(poloidal_points[:, 2])
            R = self.eq_field.eqdsk.rmagx + avg_minor_radius * np.cos(filament_etas)
            phi = filament_etas * m_local / n_local
            Z = self.eq_field.eqdsk.zmagx - avg_minor_radius * np.sin(filament_etas)
            filament_points = np.column_stack((R, phi, Z))
        elif trace_type == EquilibriumFilamentTracer.TraceType.NAIVE:
            # Follows the rational surface but not the magnetic field
            phi = filament_etas * m_local / n_local
            filament_points = np.column_stack(
                (poloidal_points[:, 0], phi, poloidal_points[:, 1])
            )
        elif trace_type == EquilibriumFilamentTracer.TraceType.AVERAGE:
            # determine d(phi)/d(eta) from magnetic field
            R = poloidal_points[:, 0]
            Z = poloidal_points[:, 1]
            B = self.eq_field.get_field_at_point(R, Z)

            # phi_k is the integral of d(phi)/dl over the segments BEFORE point k,
            # so phi_0 = 0 exactly and no segment is shifted
            # d(phi)/dl is evaluated at the points and integrated per segment with the trapezoid rule (second order).
            dl = np.sqrt(np.diff(R) ** 2 + np.diff(Z) ** 2)
            dphi_dl = _d_phi_dl(1.0, R, np.sqrt(B[0] ** 2 + B[2] ** 2), B[1])
            d_phi = self.helicity_sign * dl * 0.5 * (dphi_dl[:-1] + dphi_dl[1:])
            phi = np.concatenate([[0.0], np.cumsum(d_phi)])

            # sign_Bt carries the direction of the toroidal field, helicity_sign whether
            # trace is parallel (+1) or antiparallel (-1) to the field.
            known_phi_end = self.helicity_sign * sign_Bt * 2 * np.pi * m_local / n_local
            q_eff = np.abs(phi[-1]) / (2 * np.pi)
            closure_error = np.abs(phi[-1] - known_phi_end) / np.abs(known_phi_end)
            logger.debug(
                f"m={self.m} n={self.n}: psi_q={psi_q:.6f}, q_requested={m_local / n_local:.4f}, "
                f"q_eff={q_eff:.4f}, closure_error={closure_error:.4f}"
            )
            if closure_error > self.closure_rtol:
                raise FilamentTraceError(
                    f"m={self.m} n={self.n}: field line q_eff={q_eff:.4f} misses "
                    f"q={m_local / n_local:.4f} by {100 * closure_error:.1f} percent "
                    f"(closure_rtol={self.closure_rtol}), phi_end={phi[-1]:.4f} "
                    f"expected {known_phi_end:.4f}"
                )

            # Numerical correction so the final point is at exactly 2 pi m / n
            phi = phi * (known_phi_end / phi[-1])

            filament_points = np.column_stack((R, phi, Z))
        else:
            raise ValueError("Unknown tracing trace_type")

        # Cache the traced filament
        self.trace_cache[key] = (filament_points, filament_etas)

        return filament_points, filament_etas
