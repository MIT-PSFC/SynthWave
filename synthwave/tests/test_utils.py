import pytest
import numpy as np


from synthwave.magnetic_geometry.utils import (
    cylindrical_to_cartesian,
    cartesian_to_cylindrical,
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
