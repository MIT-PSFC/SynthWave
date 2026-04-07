"""Tests for EquilibriumField methods."""

import os
import unittest.mock as mock

import freeqdsk
import numpy as np
import pytest

from synthwave import PACKAGE_ROOT
from synthwave.magnetic_geometry.equilibrium_field import EquilibriumField


@pytest.fixture(scope="module")
def eqdsk():
    eqdsk_file = os.path.join(PACKAGE_ROOT, "input_data", "cmod", "g1051202011.1000")
    with open(eqdsk_file, "r") as f:
        return freeqdsk.geqdsk.read(f)


@pytest.mark.parametrize(
    "q_target",
    [3 / 2, 2 / 1, 3 / 1, 4 / 3, 5 / 4],
)
def test_get_psi_of_q_returns_correct_q(q_target, eqdsk):
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
def test_get_psi_of_q_in_range(q_target, eqdsk):
    """Returned psi must lie within psi_grid bounds."""
    eq_field = EquilibriumField(eqdsk)
    psi = eq_field.get_psi_of_q(q_target)
    psi_min = min(eq_field.psi_grid[0], eq_field.psi_grid[-1])
    psi_max = max(eq_field.psi_grid[0], eq_field.psi_grid[-1])
    assert psi_min <= psi <= psi_max


@pytest.mark.parametrize(
    "q_target",
    [3 / 2, 2 / 1, 3 / 1, 4 / 3, 5 / 4],
)
def test_core_psi_consistency_check_in_range_returns_psi(q_target, eqdsk):
    """When psi is already in range, the check must return it unchanged."""
    eq_field = EquilibriumField(eqdsk)
    qpsi_grid = eq_field.qpsi(eq_field.psi_grid)
    psi_mid = (eq_field.psi_grid[0] + eq_field.psi_grid[-1]) / 2
    result = eq_field.core_psi_consistency_check(qpsi_grid, psi_mid, q_target)
    assert result == psi_mid


def test_core_psi_consistency_check_out_of_range_raises(eqdsk):
    """When psi is far outside range, ValueError must be raised."""
    eq_field = EquilibriumField(eqdsk)
    qpsi_grid = eq_field.qpsi(eq_field.psi_grid)
    # Use a psi value above psi_grid[-1] (past the LCFS for increasing grid)
    psi_far = eq_field.psi_grid[-1] + abs(eq_field.psi_grid[-1]) * 10
    with pytest.raises(ValueError, match="outside the gEQDSK range"):
        eq_field.core_psi_consistency_check(qpsi_grid, psi_far, 100.0)


def test_core_psi_consistency_check_monotonic(eqdsk):
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
    decreasing_q[-2] = decreasing_q[-1] - 1e-4  # create a small plateau near the axis
    assert np.all(np.diff(decreasing_psi) < 0), "decreasing_psi must be decreasing"

    # Patch the equilibrium field's psi_grid with our synthetic decreasing grid
    with mock.patch.object(eq_field, "psi_grid", decreasing_psi):
        # A psi well inside the decreasing range should pass through unchanged
        psi_mid = (decreasing_psi[0] + decreasing_psi[-1]) / 2
        result = eq_field.core_psi_consistency_check(decreasing_q, psi_mid, 2.0)
        assert result == psi_mid

        # A psi too low (past axis) should be fixed if possible
        psi_too_low = decreasing_psi[-1] - abs(decreasing_psi[-1]) * 0.5
        psi_fixed = eq_field.core_psi_consistency_check(decreasing_q, psi_too_low, 0.1)
        assert decreasing_psi[-1] <= psi_fixed <= decreasing_psi[0], (
            "Fixed psi must be within psi_grid bounds"
        )

        # A psi too high (past LCFS) must raise
        psi_too_high = decreasing_psi[0] + abs(decreasing_psi[0]) * 0.5
        with pytest.raises(ValueError, match="outside the gEQDSK range"):
            eq_field.core_psi_consistency_check(decreasing_q, psi_too_high, 100.0)
