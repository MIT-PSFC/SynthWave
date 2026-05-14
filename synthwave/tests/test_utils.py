import numpy as np
import pytest

from synthwave.magnetic_geometry.utils import (
    angle_domain,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    wrapped_diff,
)


@pytest.mark.parametrize(
    ["cylindrical_coords", "expected_cartesian_coords"],
    [
        ([1, 0, 0], [1.0, 0.0, 0.0]),  # R=1, phi=0, Z=0
        ([1, np.pi / 2, 0], [0.0, 1.0, 0.0]),  # R=1, phi=pi/2, Z=0
        ([2, np.pi, 1], [-2.0, 0.0, 1.0]),  # R=2, phi=pi, Z=1
        (
            [3, np.pi / 4, -1],
            [3 / np.sqrt(2), 3 / np.sqrt(2), -1],
        ),  # R=3, phi=pi/4, Z=-1
        ([1, 3 * np.pi / 2, 2], [0.0, -1.0, 2.0]),  # R=1, phi=3*pi/2, Z=2
    ],
)
def test_cylindrical_to_cartesian(cylindrical_coords, expected_cartesian_coords):
    R, phi, Z = cylindrical_coords
    expected_x, expected_y, expected_z = expected_cartesian_coords
    cartesian_coords = cylindrical_to_cartesian(R, phi, Z)
    assert np.allclose(cartesian_coords, [expected_x, expected_y, expected_z])


@pytest.mark.parametrize(
    ["cartesian_coords", "expected_cylindrical_coords"],
    [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),  # x=1, y=0, z=0
        ([0.0, 1.0, 0.0], [1.0, np.pi / 2, 0.0]),  # x=0, y=1, z=0
        ([-2.0, 0.0, 1.0], [2.0, np.pi, 1.0]),  # x=-2, y=0, z=1
        (
            [3 / np.sqrt(2), 3 / np.sqrt(2), -1],
            [3.0, np.pi / 4, -1.0],
        ),  # x=y=3/sqrt(2), z=-1
        ([0.0, -1.0, 2.0], [1.0, -np.pi / 2, 2.0]),  # x=0, y=-1, z=2
    ],
)
def test_cartesian_to_cylindrical(cartesian_coords, expected_cylindrical_coords):
    x, y, z = cartesian_coords
    expected_R, expected_phi, expected_Z = expected_cylindrical_coords
    cylindrical_coords = cartesian_to_cylindrical(x, y, z)
    assert np.allclose(cylindrical_coords, [expected_R, expected_phi, expected_Z])


def test_angle_domain():
    # Format: [input_angle, expected_output_angle]
    angles_expected = np.array(
        [
            [-3.00001 * np.pi, np.pi],  # wrap to pi
            [-2.99999 * np.pi, -np.pi],  # stay at -pi
            [-2 * np.pi, 0],  # Far from boundary
            [0, 0],  # Far from boundary
            [2 * np.pi, 0],  # Far from boundary
            [2.99999 * np.pi, np.pi],  # Stay at pi
            [3.00001 * np.pi, -np.pi],  # Wrap to -pi
        ]
    )
    assert np.allclose(
        angle_domain(angles_expected[:, 0]), angles_expected[:, 1], atol=1e-4
    )


def test_wrapped_diff():
    # Format: [phase1, phase2, expected_diff]
    test_cases = np.array(
        [
            [0, 0, 0],  # No difference
            [np.pi / 2, 0, np.pi / 2],  # Simple case
            [0, np.pi / 2, -np.pi / 2],  # Simple case
            [np.pi, -np.pi, 0],  # Wrap around boundary
            [-3 * np.pi / 4, 3 * np.pi / 4, np.pi / 2],  # Wrap around boundary
            [3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 2],  # Wrap around boundary
        ]
    )
    assert np.allclose(
        wrapped_diff(test_cases[:, 0], test_cases[:, 1]), test_cases[:, 2], atol=1e-4
    )
