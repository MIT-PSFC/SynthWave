"""Tests for EquilibriumField methods."""

import os
import unittest.mock as mock

import freeqdsk
import netCDF4  # noqa: F401 - must precede any OpenFUSIONToolkit import to avoid HDF5 conflict
import numpy as np
import pytest

from synthwave import PACKAGE_ROOT
from synthwave.magnetic_geometry.equilibrium_field import (
    EquilibriumField,
    convert_cocos,
    detect_cocos,
)


@pytest.fixture(scope="module")
def eqdsk():
    eqdsk_file = os.path.join(PACKAGE_ROOT, "input_data", "cmod", "g1051202011.1000")
    with open(eqdsk_file, "r") as f:
        return freeqdsk.geqdsk.read(f)


class TestEquilibriumField:
    class TestGetPsiOfQ:
        """Tests for EquilibriumField.get_psi_of_q."""

        @pytest.mark.parametrize(
            "q_target",
            [3 / 2, 2 / 1, 3 / 1, 4 / 3, 5 / 4],
        )
        def test_get_psi_of_q_returns_correct_q(self, q_target, eqdsk):
            """get_psi_of_q must return a psi such that qpsi(psi) == q_target."""
            eq_field = EquilibriumField(eqdsk)
            psi = eq_field.get_psi_of_q(q_target)
            q_recovered = float(eq_field.qpsi(psi))
            assert np.isclose(q_recovered, q_target, atol=1e-2), (
                f"Expected q={q_target}, got q={q_recovered} at psi={psi}"
            )

        @pytest.mark.parametrize(
            "q_target",
            [3 / 2, 2 / 1, 3 / 1, 4 / 3, 5 / 4],
        )
        def test_get_psi_of_q_in_range(self, q_target, eqdsk):
            """Returned psi must lie within psi_grid bounds."""
            eq_field = EquilibriumField(eqdsk)
            psi = eq_field.get_psi_of_q(q_target)
            psi_min = min(eq_field.psi_grid[0], eq_field.psi_grid[-1])
            psi_max = max(eq_field.psi_grid[0], eq_field.psi_grid[-1])
            assert psi_min <= psi <= psi_max

    class TestCorePsiConsistencyCheck:
        @pytest.mark.parametrize(
            "q_target",
            [3 / 2, 2 / 1, 3 / 1, 4 / 3, 5 / 4],
        )
        def test_core_psi_consistency_check_in_range_returns_psi(self, q_target, eqdsk):
            """When psi is already in range, the check must return it unchanged."""
            eq_field = EquilibriumField(eqdsk)
            qpsi_grid = eq_field.qpsi(eq_field.psi_grid)
            psi_mid = (eq_field.psi_grid[0] + eq_field.psi_grid[-1]) / 2
            result = eq_field.core_psi_consistency_check(qpsi_grid, psi_mid, q_target)
            assert result == psi_mid

        def test_core_psi_consistency_check_out_of_range_raises(self, eqdsk):
            """When psi is far outside range, ValueError must be raised."""
            eq_field = EquilibriumField(eqdsk)
            qpsi_grid = eq_field.qpsi(eq_field.psi_grid)
            # Use a psi value above psi_grid[-1] (past the LCFS for increasing grid)
            psi_far = eq_field.psi_grid[-1] + abs(eq_field.psi_grid[-1]) * 10
            with pytest.raises(ValueError, match="outside the gEQDSK range"):
                eq_field.core_psi_consistency_check(qpsi_grid, psi_far, 100.0)

        def test_core_psi_consistency_check_monotonic(self, eqdsk):
            """Boundary check works correctly for an increasing and synthetic decreasing psi_grid."""
            eq_field = EquilibriumField(eqdsk)

            # Test with the default increasing psi_grid and a qpsi_grid that's increasing but has a plateau near the axis
            qpsi_grid = eq_field.qpsi(eq_field.psi_grid)
            qpsi_grid[1] = qpsi_grid[0] + 1e-4  # create a small plateau near the axis

            # A psi in range should pass through unchanged
            psi_mid = (eq_field.psi_grid[0] + eq_field.psi_grid[-1]) / 2
            result = eq_field.core_psi_consistency_check(qpsi_grid, psi_mid, 2.0)
            assert result == psi_mid

            # A psi too low (past axis) should be fixed if possible
            psi_too_low = eq_field.psi_grid[0] - abs(eq_field.psi_grid[0]) * 0.5
            psi_fixed = eq_field.core_psi_consistency_check(qpsi_grid, psi_too_low, 0.1)
            assert eq_field.psi_grid[0] <= psi_fixed <= eq_field.psi_grid[-1], (
                "Fixed psi must be within psi_grid bounds"
            )

            # A psi too high (past LCFS) must raise
            psi_too_high = eq_field.psi_grid[-1] + abs(eq_field.psi_grid[-1]) * 0.5
            with pytest.raises(ValueError, match="outside the gEQDSK range"):
                eq_field.core_psi_consistency_check(qpsi_grid, psi_too_high, 100.0)

            # Reverse to create a genuinely decreasing psi_grid and qpsi_grid (index 0 at LCFS, index -1 at axis)
            decreasing_psi = eq_field.psi_grid[::-1]
            decreasing_q = eq_field.qpsi(eq_field.psi_grid)[::-1]
            decreasing_q[-2] = (
                decreasing_q[-1] - 1e-4
            )  # create a small plateau near the axis
            assert np.all(np.diff(decreasing_psi) < 0), (
                "decreasing_psi must be decreasing"
            )

            # Patch the equilibrium field's psi_grid with our synthetic decreasing grid
            with mock.patch.object(eq_field, "psi_grid", decreasing_psi):
                # A psi well inside the decreasing range should pass through unchanged
                psi_mid = (decreasing_psi[0] + decreasing_psi[-1]) / 2
                result = eq_field.core_psi_consistency_check(decreasing_q, psi_mid, 2.0)
                assert result == psi_mid

                # A psi too low (past axis) should be fixed if possible
                psi_too_low = decreasing_psi[-1] - abs(decreasing_psi[-1]) * 0.5
                psi_fixed = eq_field.core_psi_consistency_check(
                    decreasing_q, psi_too_low, 0.1
                )
                assert decreasing_psi[-1] <= psi_fixed <= decreasing_psi[0], (
                    "Fixed psi must be within psi_grid bounds"
                )

                # A psi too high (past LCFS) must raise
                psi_too_high = decreasing_psi[0] + abs(decreasing_psi[0]) * 0.5
                with pytest.raises(ValueError, match="outside the gEQDSK range"):
                    eq_field.core_psi_consistency_check(
                        decreasing_q, psi_too_high, 100.0
                    )


class TestCocos:
    @pytest.fixture(scope="class")
    def eqdsk_cmod(self):
        """Get the C-Mod eqdsk"""
        eqdsk_file = os.path.join(
            PACKAGE_ROOT, "input_data", "cmod", "g1051202011.1000"
        )
        with open(eqdsk_file, "r") as f:
            return freeqdsk.geqdsk.read(f)

    @pytest.fixture(scope="class")
    def eqdsk_d3d(self):
        eqdsk_file = os.path.join(
            PACKAGE_ROOT,
            "..",
            "submodules",
            "OpenFUSIONToolkit",
            "examples",
            "TokaMaker",
            "DIIID",
            "g192185.02440",
        )
        with open(eqdsk_file, "r") as f:
            eqdsk = freeqdsk.geqdsk.read(f)

        return eqdsk

    @pytest.fixture(scope="class")
    def eqdsk_tcv(self):
        import xarray as xr
        from scipy.interpolate import interp1d

        fname = "/usr/local/mfe/ml_data_dump/TMDB/scratch/82878_tars_input.nc"
        tidx = 1100

        ds = xr.open_dataset(fname)
        ds_eq = ds.sel(time_idx=tidx).isel(frequency_idx=0)

        nx = len(ds_eq["rgrid"])
        ny = len(ds_eq["zgrid"])

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
            return interp1d(psi_old, arr, kind="linear")(psi_new)

        eqdsk = freeqdsk.geqdsk.GEQDSKFile(
            comment="TCV LIUQE",
            shot=82878,
            nx=nx,
            ny=ny,
            rdim=float(ds_eq["rgrid"][-1] - ds_eq["rgrid"][0]),
            zdim=float(ds_eq["zgrid"][-1] - ds_eq["zgrid"][0]),
            rcentr=float(ds_eq["rgrid"][nx // 2]),
            rleft=float(ds_eq["rgrid"][0]),
            zmid=float(ds_eq["zgrid"][ny // 2]),
            rmagx=float(ds_eq["rmaxis"]),
            zmagx=float(ds_eq["zmaxis"]),
            simagx=float(ds_eq["ssimag"]),
            sibdry=float(ds_eq["ssibry"]),
            bcentr=float(ds_eq["bcentr"]),
            cpasma=float(ds_eq["cpasma"]),
            fpol=_resample_to_nx(ds_eq["fpol"].values),
            pres=_resample_to_nx(ds_eq["pres"].values),
            ffprime=_resample_to_nx(ds_eq["ffprim"].values),
            pprime=_resample_to_nx(ds_eq["pprime"].values),
            psi=ds_eq["psirz"].values,
            qpsi=_resample_to_nx(ds_eq["qpsi"].values),
            nbdry=len(ds_eq["rbbbs"]),
            nlim=0,
            rbdry=ds_eq["rbbbs"].values,
            zbdry=ds_eq["zbbbs"].values,
            rlim=[],
            zlim=[],
        )

        return eqdsk

    class TestDetectCocos:
        def test_detect_cocos_cmod(self, eqdsk_cmod):
            """C-Mod uses EFIT convention (sigma_RphiZ=+1, e_Bp=0, psi increasing).
            COCOS depends on sign(Ip) and sign(B0)
            https://efit-ai.gitlab.io/efit/files.html
            """

            sign_Ip = np.sign(eqdsk_cmod.cpasma)
            sign_B0 = np.sign(float(eqdsk_cmod.bcentr))
            if sign_B0 == 0:
                # When bcentr=0, infer sign_B0 from fpol (assuming sign_RphiZ=+1).
                sign_B0 = np.sign(np.nanmean(eqdsk_cmod.fpol))

            if sign_Ip > 0 and sign_B0 > 0:
                expected_cocos = 1
            elif sign_Ip < 0 and sign_B0 > 0:
                expected_cocos = 3
            elif sign_Ip > 0 and sign_B0 < 0:
                expected_cocos = 5
            elif sign_Ip < 0 and sign_B0 < 0:
                expected_cocos = 7
            else:
                raise ValueError("Invalid signs for Ip and q")

            cocos = detect_cocos(eqdsk_cmod)
            assert cocos == expected_cocos, (
                f"Expected COCOS {expected_cocos} for C-Mod, got {cocos}"
            )

        def test_detect_cocos_d3d(self, eqdsk_d3d):
            """DIII-D uses EFIT convention (sigma_RphiZ=+1, e_Bp=0, psi increasing).
            COCOS depends on sign(Ip) and sign(B0)
            https://efit-ai.gitlab.io/efit/files.html
            """

            sign_Ip = np.sign(eqdsk_d3d.cpasma)
            sign_B0 = np.sign(float(eqdsk_d3d.bcentr))
            if sign_B0 == 0:
                # When bcentr=0, infer sign_B0 from fpol (assuming sign_RphiZ=+1).
                sign_B0 = np.sign(np.nanmean(eqdsk_d3d.fpol))

            if sign_Ip > 0 and sign_B0 > 0:
                expected_cocos = 1
            elif sign_Ip < 0 and sign_B0 > 0:
                expected_cocos = 3
            elif sign_Ip > 0 and sign_B0 < 0:
                expected_cocos = 5
            elif sign_Ip < 0 and sign_B0 < 0:
                expected_cocos = 7
            else:
                raise ValueError("Invalid signs for Ip and q")

            cocos = detect_cocos(eqdsk_d3d)
            assert cocos == expected_cocos, (
                f"Expected COCOS {expected_cocos} for DIII-D, got {cocos}"
            )

        @pytest.mark.skipif(
            condition=not os.path.exists(
                "/usr/local/mfe/ml_data_dump/TMDB/scratch/82878_tars_input.nc"
            ),
            reason="Test requires TCV data which is not open source",
        )
        def test_detect_cocos_tcv(self, eqdsk_tcv):
            """TCV uses LIUQE which uses COCOS 17 (e_Bp=1, sign_Bp=-1, sign_RphiZ=1, sign_rhotp=1)"""
            cocos = detect_cocos(eqdsk_tcv)
            expected_cocos = 17
            assert cocos == expected_cocos, (
                f"Expected COCOS {expected_cocos} for TCV, got {cocos}"
            )

    class TestConvertCocos:
        _TCV_FILE = "/usr/local/mfe/ml_data_dump/TMDB/scratch/82878_tars_input.nc"

        @pytest.fixture
        def eqdsk(self, request):
            return request.getfixturevalue(request.param)

        @pytest.mark.parametrize(
            "eqdsk",
            [
                "eqdsk_cmod",
                "eqdsk_d3d",
                pytest.param(
                    "eqdsk_tcv",
                    marks=pytest.mark.skipif(
                        condition=not os.path.exists(
                            "/usr/local/mfe/ml_data_dump/TMDB/scratch/82878_tars_input.nc"
                        ),
                        reason="Test requires TCV data which is not open source",
                    ),
                ),
            ],
            indirect=True,
        )
        def test_convert_cocos_same(self, eqdsk):
            """Test that converting to the same COCOS convention returns the same result"""
            cocos = detect_cocos(eqdsk)
            new_eqdsk = convert_cocos(eqdsk, cocos, cocos)

            # Check that the new_eqdsk matches the original eqdsk
            assert np.isclose(new_eqdsk.simagx, eqdsk.simagx), (
                "simagx should be unchanged"
            )
            assert np.isclose(new_eqdsk.sibdry, eqdsk.sibdry), (
                "sibdry should be unchanged"
            )
            assert np.allclose(new_eqdsk.psi, eqdsk.psi), (
                "psi array should be unchanged"
            )
            assert np.allclose(new_eqdsk.fpol, eqdsk.fpol), (
                "fpol array should be unchanged"
            )
            assert np.allclose(new_eqdsk.ffprime, eqdsk.ffprime), (
                "ffprime array should be unchanged"
            )
            assert np.allclose(new_eqdsk.pprime, eqdsk.pprime), (
                "pprime array should be unchanged"
            )

        @pytest.mark.parametrize(
            "eqdsk",
            [
                "eqdsk_cmod",
                "eqdsk_d3d",
                pytest.param(
                    "eqdsk_tcv",
                    marks=pytest.mark.skipif(
                        condition=not os.path.exists(
                            "/usr/local/mfe/ml_data_dump/TMDB/scratch/82878_tars_input.nc"
                        ),
                        reason="Test requires TCV data which is not open source",
                    ),
                ),
            ],
            indirect=True,
        )
        def test_convert_cocos_internal(self, eqdsk):
            """Test that converting to COCOS 1 (used internally) creates an equilibrium that has some expected properties."""
            cocos_input = detect_cocos(eqdsk)
            cocos_target = 1  # COCOS 1 is the convention used internally by SynthWave
            eqdsk_cocos1 = convert_cocos(
                eqdsk, cocos_target=cocos_target, cocos_input=cocos_input
            )

            detected_cocos = detect_cocos(eqdsk_cocos1)
            assert detected_cocos == cocos_target, (
                f"Expected converted COCOS to be {cocos_target}, got {detected_cocos}"
            )

            # In COCOS 1 (sign_Bp=+1), sign(dpsi/drho) = sign(Ip) per Sauter Table III.
            # Psi increases from axis to boundary iff Ip > 0.
            sign_Ip = float(eqdsk_cocos1.cpasma)
            assert (
                float(eqdsk_cocos1.sibdry) - float(eqdsk_cocos1.simagx)
            ) * sign_Ip > 0, (
                f"Expected sign(sibdry - simagx) == sign(Ip) in COCOS {cocos_target}"
            )

        @pytest.mark.parametrize(
            "eqdsk",
            [
                "eqdsk_cmod",
                "eqdsk_d3d",
                pytest.param(
                    "eqdsk_tcv",
                    marks=pytest.mark.skipif(
                        condition=not os.path.exists(
                            "/usr/local/mfe/ml_data_dump/TMDB/scratch/82878_tars_input.nc"
                        ),
                        reason="Test requires TCV data which is not open source",
                    ),
                ),
            ],
            indirect=True,
        )
        def test_convert_cocos_same_through_internal(self, eqdsk):
            """Test that converting to COCOS 1 (used internally) and back to the original
            creates an identical equilibrium"""
            cocos_input = detect_cocos(eqdsk)
            cocos_target = 1  # COCOS 1 is the convention used internally by SynthWave
            eqdsk_cocos1 = convert_cocos(
                eqdsk, cocos_target=cocos_target, cocos_input=cocos_input
            )
            eqdsk_converted_back = convert_cocos(
                eqdsk_cocos1, cocos_target=cocos_input, cocos_input=cocos_target
            )

            # Check that the converted_back eqdsk matches the original eqdsk
            assert np.isclose(eqdsk_converted_back.simagx, eqdsk.simagx), (
                "simagx should be unchanged after round-trip conversion"
            )
            assert np.isclose(eqdsk_converted_back.sibdry, eqdsk.sibdry), (
                "sibdry should be unchanged after round-trip conversion"
            )
            assert np.allclose(eqdsk_converted_back.psi, eqdsk.psi), (
                "psi array should be unchanged after round-trip conversion"
            )
            assert np.allclose(eqdsk_converted_back.fpol, eqdsk.fpol), (
                "fpol array should be unchanged after round-trip conversion"
            )
            assert np.allclose(eqdsk_converted_back.ffprime, eqdsk.ffprime), (
                "ffprime array should be unchanged after round-trip conversion"
            )
            assert np.allclose(eqdsk_converted_back.pprime, eqdsk.pprime), (
                "pprime array should be unchanged after round-trip conversion"
            )
