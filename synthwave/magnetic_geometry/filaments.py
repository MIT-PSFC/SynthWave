import xarray as xr
from typing import Optional
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import newton
from fractions import Fraction
from synthwave.magnetic_geometry.utils import cylindrical_to_cartesian

from synthwave.magnetic_geometry.equilibrium_field import (
    EquilibriumField,
)


class FilamentTracer(ABC):
    """Abstract class for filament representation."""

    def __init__(self, m: int, n: int, num_points: Optional[int] = 1000):
        self.m = m
        self.n = n
        self.num_points = num_points
        self.traces = {}  # Dict to store traces of different numbers of points

    @abstractmethod
    def _trace(
        self, num_filament_points: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace the filament and return the points in cylindrical coordinates (R, phi, Z) as well as the corresponding eta values.

        Args:
            num_filament_points (Optional[int]): Number of points to trace along the filament. If None, uses self.num_points.
        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - filament_points: Array of shape (N, 3) where each row is [R, phi, Z] in cylindrical coordinates.
                - filament_etas: Array of shape (N,) containing the eta values corresponding to each point.
        """

    def trace(
        self, num_filament_points: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace the filament and return the points in cylindrical coordinates (R, phi, Z), and the corresponding eta values."""
        if num_filament_points in self.traces:
            filament_points, filament_etas = self.traces[num_filament_points]
        else:
            filament_points, filament_etas = self._trace(num_filament_points)
            self.traces[num_filament_points] = (filament_points, filament_etas)

        return filament_points, filament_etas

    def get_filament_ds(
        self,
        num_filaments: int,
        num_filament_points: Optional[int] = None,
        coordinate_system: Optional[str] = "cartesian",
    ) -> xr.Dataset:
        """Generate points and corresponding currents for multiple filaments.

        Args:
            num_filaments (int): How many individual filaments to create
            num_filament_points (Optional[int], default = None): Number of points per filament. If None, uses self.num_points
            coordinate_system (Optional[str], default = "cartesian"): Coordinate system for output points. Options are "cylindrical", "cartesian", or "toroidal".

        Returns:
            xr.Dataset: Dataset containing filament points and currents. Dimensions are 'filament' and 'point', with variables 'R', 'phi', 'Z' or 'x', 'y', 'z' or 'eta', 'phi', and 'current'.
        """

        if num_filaments <= 0:
            raise ValueError("num_filaments must be a positive integer")

        if num_filament_points is None:
            num_filament_points = self.num_points

        if coordinate_system not in ["cylindrical", "cartesian", "toroidal"]:
            raise ValueError(
                "coordinate_system must be either 'cylindrical', 'cartesian', or 'toroidal'"
            )

        # Start with a filament that has zero toroidal offset
        base_filament_points, filament_etas = self.trace(num_filament_points)

        # Create toroidal offsets and corresponding currents
        starting_angles = np.linspace(0, 2 * np.pi, num_filaments, endpoint=False)

        all_filament_points = np.repeat(
            base_filament_points[np.newaxis, :, :], num_filaments, axis=0
        )  # Shape (num_filaments, N, 3)
        all_filament_points[:, :, 1] += starting_angles[
            :, np.newaxis
        ]  # Apply toroidal offsets

        # Complex currents for rotating wave: I(phi) = I_0 * exp(i*n*phi)
        ratio = Fraction(self.m, self.n)
        n_local = ratio.denominator
        filament_currents = np.exp(1j * starting_angles * n_local)

        if coordinate_system == "cylindrical":
            ds = xr.Dataset(
                data_vars={
                    "R": (("filament", "point"), all_filament_points[:, :, 0]),
                    "phi": (("filament", "point"), all_filament_points[:, :, 1]),
                    "Z": (("filament", "point"), all_filament_points[:, :, 2]),
                    "current": (("filament"), filament_currents),
                },
                coords={
                    "filament": np.arange(num_filaments),
                    "point": np.arange(num_filament_points),
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
                    "current": (("filament"), filament_currents),
                },
                coords={
                    "filament": np.arange(num_filaments),
                    "point": np.arange(num_filament_points),
                },
            )
        elif coordinate_system == "toroidal":
            ds = xr.Dataset(
                data_vars={
                    "eta": (("point"), filament_etas),
                    "phi": (("filament", "point"), all_filament_points[:, :, 1]),
                    "current": (("filament"), filament_currents),
                },
                coords={
                    "filament": np.arange(num_filaments),
                    "point": np.arange(num_filament_points),
                },
            )

        return ds

    def get_filament_list(
        self,
        num_filaments: int,
        num_filament_points: Optional[int] = None,
        coordinate_system: str = "cartesian",
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate a list of filaments, each represented as an array of shape (N, 3) in cylindrical coordinates.

        Args:
            num_filaments (int): Number of filaments to generate.
            num_filament_points (Optional[int], default = None): Number of points per filament. If None, uses self.num_points.
            coordinate_system (str, default = "cartesian"): Coordinate system for output points. Options are "cylindrical" or "cartesian".

        Returns:
            list[np.ndarray]: List of filaments, each of shape (N, 3) with columns [x, y, z] or [R, phi, Z].
            list[float]: List of filament currents.
        """
        if num_filaments <= 0:
            raise ValueError("num_filaments must be a positive integer")

        if num_filament_points is None:
            num_filament_points = self.num_points

        if coordinate_system not in ["cylindrical", "cartesian"]:
            raise ValueError(
                "coordinate_system must be either 'cylindrical' or 'cartesian'"
            )

        filament_points_ds = self.get_filament_ds(
            num_filaments, num_filament_points, coordinate_system
        )

        filament_list = []
        for i in range(num_filaments):
            if coordinate_system == "cylindrical":
                R = filament_points_ds["R"].isel(filament=i).values
                phi = filament_points_ds["phi"].isel(filament=i).values
                Z = filament_points_ds["Z"].isel(filament=i).values
                filament_array = np.array([R, phi, Z]).T
            elif coordinate_system == "cartesian":
                x = filament_points_ds["x"].isel(filament=i).values
                y = filament_points_ds["y"].isel(filament=i).values
                z = filament_points_ds["z"].isel(filament=i).values
                filament_array = np.array([x, y, z]).T
            filament_list.append(filament_array)

        current_list = filament_points_ds["current"].values.tolist()

        return filament_list, current_list

    def make_filament_spline(self):
        """Create a spline which puts eta in terms of phi for this filament."""
        filament_points, filament_etas = self._trace(self.num_points)

        spline = make_interp_spline(filament_points[:, 1], filament_etas)
        return spline


class ToroidalFilamentTracer(FilamentTracer):
    """Filament for toroidal approximation of the magnetic geometry."""

    def __init__(
        self,
        m: int,
        n: int,
        R0: float,
        Z0: float,
        a: float,
        num_points: Optional[int] = 1000,
    ):
        """Initialize a toroidal filament with a circular cross-section.

        Parameters
        ----------
        m : int
            Poloidal mode number
        n : int
            Toroidal mode number
        R0 : float
            Major radius of the magnetic axis
        Z0 : float
            Vertical position of the magnetic axis
        a : float
            Minor radius of the circular cross-section
        num_points : int, optional
            Number of points to trace around the filament

        """
        super().__init__(m, n, num_points)
        self.R0 = R0
        self.Z0 = Z0
        self.a = a

    def _trace(
        self, num_filament_points: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_filament_points is None:
            num_filament_points = self.num_points

        # Create a circular filament around the magnetic axis
        phi = np.linspace(0, 2 * np.pi * self.m / self.n, int(num_filament_points))
        filament_etas = np.linspace(0, 2 * np.pi, int(num_filament_points))
        R = self.R0 + self.a * np.cos(filament_etas)
        Z = self.Z0 + self.a * np.sin(filament_etas)

        filament_points = np.column_stack((R, phi, Z))

        return filament_points, filament_etas


class EquilibriumFilamentTracer(FilamentTracer):
    """Filament traced along an equilibrium magnetic field."""

    class TraceType(Enum):
        CYLINDRICAL = 0  # Cylindrical approximation of the magnetic geometry
        NAIVE = 1  # Naive tracing, following the rational surface but not the field
        SINGLE = 2  # Single tracing method, using the magnetic field to determine d(phi)/d(eta)
        AVERAGE = 3  # Average tracing method, using the magnetic field to determine d(phi)/d(eta) and averaging between points

    def __init__(
        self,
        m: int,
        n: int,
        eq_field: EquilibriumField,
        num_points: Optional[int] = 1000,
    ):
        """Initialize an equilibrium filament.

        Parameters
        ----------
        m : int
            Poloidal mode number
        n : int
            Toroidal mode number
        eq_field : EquilibriumField
            EquilibriumField object containing the magnetic field data
        num_points : int, optional
            Number of points to trace around the filament

        """
        super().__init__(m, n, int(num_points))
        self.eq_field = eq_field

    def _trace(
        self,
        num_filament_points: Optional[int] = None,
        method: TraceType = TraceType.SINGLE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace a filament"""
        if num_filament_points is None:
            num_filament_points = self.num_points

        # Correction for m/n as integer multiples (otherwise leads to ``wandering'' filaments)
        ratio = Fraction(self.m, self.n)
        m_local = ratio.numerator
        n_local = ratio.denominator
        psi_q = self.eq_field.get_psi_of_q(m_local / n_local)

        filament_etas = np.linspace(0, 2 * np.pi, num_filament_points)
        poloidal_points = np.zeros((num_filament_points, 3))  # R, Z, a

        # Start at the outboard midplane, slightly outside magnetic axis
        Z_start = self.eq_field.eqdsk.zmagx
        R_guess = self.eq_field.eqdsk.rmagx + 0.1
        R_start = newton(
            func=lambda R: self.eq_field.psi.ev(R, Z_start) - psi_q,
            x0=R_guess,
            fprime=lambda R: self.eq_field.psi.ev(R, Z_start, dx=1, dy=0),
            maxiter=800,
            tol=1e-3,
        )

        # Sliding along minor radius a to meet the rational surface
        def _R_a(eta, a):
            R = self.eq_field.eqdsk.rmagx + (a * np.cos(eta))
            return R

        def _Z_a(eta, a):
            Z = self.eq_field.eqdsk.zmagx - (a * np.sin(eta))
            return Z

        def psi_prime_a(eta, a):
            # Derivative of psi with respect to a at a given eta
            R = _R_a(eta, a)
            Z = _Z_a(eta, a)
            return self.eq_field.psi.ev(R, Z, dx=1, dy=0) * np.cos(
                eta
            ) - self.eq_field.psi.ev(R, Z, dx=0, dy=1) * np.sin(eta)

        for i, eta in enumerate(filament_etas):
            if i == 0:
                R_prev = R_start
                Z_prev = Z_start
            else:
                R_prev = poloidal_points[i - 1, 0]
                Z_prev = poloidal_points[i - 1, 1]
            a_guess = np.sqrt(
                (R_prev - self.eq_field.eqdsk.rmagx) ** 2
                + (Z_prev - self.eq_field.eqdsk.zmagx) ** 2
            )
            a_next = newton(
                func=lambda a: self.eq_field.psi.ev(_R_a(eta, a), _Z_a(eta, a)) - psi_q,
                x0=a_guess,
                fprime=lambda a: psi_prime_a(eta, a),
            )
            poloidal_points[i, :] = [_R_a(eta, a_next), _Z_a(eta, a_next), a_next]

        def _d_phi(r, R, Bp, Bt, d_eta):
            # https://wiki.fusion.ciemat.es/wiki/Rotational_transform
            return (Bt * r * d_eta) / (R * Bp)

        # Finalize filament trace based on method
        if method == EquilibriumFilamentTracer.TraceType.CYLINDRICAL:
            # Circular cross section around the magnetic axis
            avg_minor_radius = np.mean(poloidal_points[:, 2])
            R = self.eq_field.eqdsk.rmagx + avg_minor_radius * np.cos(filament_etas)
            phi = filament_etas * m_local / n_local
            Z = self.eq_field.eqdsk.zmagx - avg_minor_radius * np.sin(filament_etas)
            filament_points = np.column_stack((R, phi, Z))
        elif method == EquilibriumFilamentTracer.TraceType.NAIVE:
            # Follows the rational surface but not the magnetic field
            phi = filament_etas * m_local / n_local
            filament_points = np.column_stack(
                (poloidal_points[:, 0], phi, poloidal_points[:, 1])
            )
        elif method in [
            EquilibriumFilamentTracer.TraceType.SINGLE,
            EquilibriumFilamentTracer.TraceType.AVERAGE,
        ]:
            # determine d(phi)/d(eta) from magnetic field
            R = poloidal_points[:, 0]
            Z = poloidal_points[:, 1]
            r = poloidal_points[:, 2]
            B = self.eq_field.get_field_at_point(R, Z)
            d_eta = np.mean(np.diff(filament_etas))
            d_phi = _d_phi(r, R, np.sqrt(B[0] ** 2 + B[2] ** 2), B[1], d_eta)
            if method == EquilibriumFilamentTracer.TraceType.SINGLE:
                phi = np.cumsum(d_phi) - d_phi[0]
            else:
                # Average d_phi between adjacent points
                d_phi_avg = (d_phi + np.roll(d_phi, -1)) / 2
                phi = np.cumsum(d_phi_avg) - d_phi_avg[0]

            # Numerical correction to ensure final point is at the proper angle
            # We can do this multiple times, whenever we know for sure that the phi is a multiple of pi * m / n

            # RNC EDIT: Sign flip necessary to account for helicity direction,
            # so known_phis matches the sign of the traced phi values
            known_phis = np.linspace(
                0, 2 * np.pi * m_local / n_local, (2 * n_local) + 1
            ) * np.sign(phi[-1])
            for i, known_phi_start in enumerate(known_phis[:-1]):
                known_phi_end = known_phis[i + 1]
                if np.sign(phi[-1]) == 1:
                    phi_indices = np.squeeze(
                        np.where((phi >= known_phi_start) & (phi <= known_phi_end))
                    )
                else:
                    phi_indices = np.squeeze(
                        np.where((phi <= known_phi_start) & (phi >= known_phi_end))
                    )

                actual_phi_start = phi[phi_indices[0]]
                actual_phi_end = phi[phi_indices[-1]]
                correction_factor = (known_phi_end - known_phi_start) / (
                    actual_phi_end - actual_phi_start
                )
                phi[phi_indices] = (
                    known_phi_start
                    + (phi[phi_indices] - actual_phi_start) * correction_factor
                )

            filament_points = np.column_stack((R, phi, Z))
        else:
            raise ValueError("Unknown tracing method")

        return filament_points, filament_etas
