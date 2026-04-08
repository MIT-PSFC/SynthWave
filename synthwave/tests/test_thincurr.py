import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xarray as xr
from OpenFUSIONToolkit import OFT_env
from sympy import nextprime

from synthwave import PACKAGE_ROOT
from synthwave.magnetic_geometry.filaments import ToroidalFilamentTracer
from synthwave.magnetic_geometry.utils import (
    angle_domain,
    create_torus_mesh,
    wrapped_diff,
)
from synthwave.mirnov.prep_thincurr_input import (
    gen_OFT_filament_and_eta_file,
    gen_OFT_sensors_file,
)
from synthwave.mirnov.run_thincurr_model import (
    calc_direct_response,
    calc_frequency_response,
)

# All tests in this file use the OpenFUSIONToolkit C++ library which has
# global state (OFT_env, ThinCurr) that cannot be shared across concurrent
# workers.  Mark the entire module serial so the conftest fixture forces
# sequential execution and GC cleanup between tests.
pytestmark = pytest.mark.serial


# Fixture for oft environment so only one is created for all tests
@pytest.fixture(scope="session")
def oft_env_fixture():
    oft_env = OFT_env(nthreads=1)
    return oft_env


@pytest.mark.parametrize(
    "mode",
    [
        {"m": 2, "n": 1},
        {"m": 3, "n": 2},
        {"m": -3, "n": 2},
        {"m": 3, "n": 1},
        {"m": 4, "n": 3},
    ],
    ids=["m2n1", "m3n2", "m-3n2", "m3n1", "m4n3"],
)
@pytest.mark.parametrize("major_radius", [1, 10])
def test_toroidal_angles(mode, major_radius, oft_env_fixture):
    tolerance_tight = np.deg2rad(0.5)
    tolerance_loose = np.deg2rad(
        2
    )  # Any more than this and I fear spectral analysis will struggle

    minor_radius_vessel = 0.35
    minor_radius_sensor = 0.34
    minor_radius_plasma = 0.3

    num_filaments = nextprime(64)
    resistivity = 1e-6
    base_num_points = 1000 * major_radius
    sensor_details = xr.Dataset(
        data_vars={
            "position": (
                ("sensor", "coord"),
                np.array(
                    [
                        [
                            major_radius + minor_radius_sensor,
                            0.0,
                            0.0,
                        ],  # sensor on x axis
                        [
                            0,
                            major_radius + minor_radius_sensor,
                            0.0,
                        ],  # sensor on y axis
                        [
                            major_radius,
                            0,
                            minor_radius_sensor,
                        ],  # sensor up top on x axis
                        [
                            0,
                            major_radius,
                            minor_radius_sensor,
                        ],  # sensor up top on y axis
                    ]
                ),
            ),
            "normal": (
                ("sensor", "coord"),
                np.array(
                    [
                        # Normals pointing radially outward
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
            ),
            "radius": ("sensor", np.array([0.01, 0.01, 0.01, 0.01])),
        },
        coords={
            "sensor": np.array(["sensor_a", "sensor_b", "sensor_c", "sensor_d"]),
        },
        attrs={
            "sensor_set_name": "test_sensors",
        },
    )

    cache_dir = os.path.join(
        PACKAGE_ROOT, "tests", "test_thincurr", "test_toroidal_angles"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Make vessel mesh and cache it since it doesn't depend on filament geometry
    torus_mesh_file = os.path.join(cache_dir, f"vessel_R{major_radius}.h5")
    if not os.path.exists(torus_mesh_file):
        torus_mesh = create_torus_mesh(
            major_radius, minor_radius_vessel, ntheta=64, nphi=128
        )
        torus_mesh.write_to_file(torus_mesh_file)

    vessel_cache_path = os.path.join(cache_dir, f"vessel_R{major_radius}.save")
    sensor_cache_path = os.path.join(
        cache_dir, f"Msensor_m{mode['m']}_n{mode['n']}_R{major_radius}.save"
    )
    mcoil_cache_path = os.path.join(
        cache_dir, f"Mcoil_m{mode['m']}_n{mode['n']}_R{major_radius}.save"
    )

    with tempfile.TemporaryDirectory() as working_directory:
        toroidal_tracer = ToroidalFilamentTracer(
            mode["m"],
            mode["n"],
            major_radius,
            0,
            minor_radius_plasma,
            base_num_points=base_num_points,
            scale_points=False,
            prevent_synthetic_structure=True,
        )
        filament_list, _ = toroidal_tracer.get_filament_list(
            num_filaments=num_filaments, coordinate_system="cartesian"
        )

        gen_OFT_filament_and_eta_file(
            working_directory=working_directory,
            filament_list=filament_list,
            resistivity_list=[resistivity],
        )

        sensor_file_path = gen_OFT_sensors_file(
            sensor_details=sensor_details,
            working_directory=working_directory,
        )

        total_response, direct_response, vessel_response = calc_frequency_response(
            oft_env=oft_env_fixture,
            tracer=toroidal_tracer,
            freq=10e3,
            mesh_file=torus_mesh_file,
            working_directory=working_directory,
            sensor_file_path=sensor_file_path,
            vessel_cache_path=vessel_cache_path,
            msensor_cache_path=sensor_cache_path,
            mcoil_cache_path=mcoil_cache_path,
        )

    # Cylindrical approximations
    toroidal_phase_ab, toroidal_phase_cd = np.pi / 2, np.pi / 2
    expected_phase_diff_ab = angle_domain(toroidal_phase_ab * mode["n"])
    expected_phase_diff_cd = angle_domain(toroidal_phase_cd * mode["n"])

    # Toroidal phase difference for the direct response should closely match cylindrical approximation
    direct_measured_phase_diff_ab = np.angle(direct_response[1] / direct_response[0])
    direct_measured_phase_diff_cd = np.angle(direct_response[3] / direct_response[2])
    assert np.isclose(
        wrapped_diff(direct_measured_phase_diff_ab, expected_phase_diff_ab),
        0,
        atol=tolerance_tight,
    ), (
        "Toroidal phase difference for direct response between sensors A and B does not match cylindrical approximation"
    )
    assert np.isclose(
        wrapped_diff(direct_measured_phase_diff_cd, expected_phase_diff_cd),
        0,
        atol=tolerance_tight,
    ), (
        "Toroidal phase difference for direct response between sensors C and D does not match cylindrical approximation"
    )

    # Poloidal phase difference for direct response may deviate due to toroidal effects
    direct_measured_phase_diff_ac = np.angle(direct_response[2] / direct_response[0])
    direct_measured_phase_diff_bd = np.angle(direct_response[3] / direct_response[1])
    assert np.isclose(
        wrapped_diff(direct_measured_phase_diff_ac, direct_measured_phase_diff_bd),
        0,
        atol=tolerance_tight,
    ), (
        "The poloidal phase difference for the direct response between sensors A and C should closely match that between sensors B and D"
    )

    # Axisymmetric vessel response should largely preserve the toroidal phase difference
    total_measured_phase_diff_ab = np.angle(total_response[1] / total_response[0])
    total_measured_phase_diff_cd = np.angle(total_response[3] / total_response[2])
    if major_radius == 1:
        assert np.isclose(
            wrapped_diff(total_measured_phase_diff_ab, expected_phase_diff_ab),
            0,
            atol=tolerance_loose,
        ), (
            "At small major radius, the toroidal phase difference for the total response between sensors A and B should still roughly match the cylindrical approximation"
        )
        assert np.isclose(
            wrapped_diff(total_measured_phase_diff_cd, expected_phase_diff_cd),
            0,
            atol=tolerance_loose,
        ), (
            "At small major radius, the toroidal phase difference for the total response between sensors C and D should still roughly match the cylindrical approximation"
        )
    else:
        assert np.isclose(
            wrapped_diff(total_measured_phase_diff_ab, expected_phase_diff_ab),
            0,
            atol=tolerance_tight,
        ), (
            "At large major radius, the toroidal phase difference for the total response between sensors A and B should match the cylindrical approximation"
        )
        assert np.isclose(
            wrapped_diff(total_measured_phase_diff_cd, expected_phase_diff_cd),
            0,
            atol=tolerance_tight,
        ), (
            "At large major radius, the toroidal phase difference for the total response between sensors C and D should match the cylindrical approximation"
        )

    # Vessel response may significantly alter the poloidal phase difference, so just check they're consistent
    total_measured_phase_diff_ac = np.angle(total_response[2] / total_response[0])
    total_measured_phase_diff_bd = np.angle(total_response[3] / total_response[1])
    assert np.isclose(
        wrapped_diff(total_measured_phase_diff_ac, total_measured_phase_diff_bd),
        0,
        atol=tolerance_tight,
    ), (
        "The poloidal phase difference for the total response between sensors A and C should roughly match that between sensors B and D"
    )


def test_gen_OFT_filament_and_eta_file():
    """gen_OFT_filament_and_eta_file must produce a valid XML file with the correct structure."""
    filament_1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    filament_2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    resistivity_list = [1e-6, 2e-6]

    with tempfile.TemporaryDirectory() as working_directory:
        gen_OFT_filament_and_eta_file(
            working_directory,
            [filament_1, filament_2],
            resistivity_list,
        )

        xml_path = os.path.join(working_directory, "oft_in.xml")
        assert os.path.exists(xml_path), "oft_in.xml was not created"

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Top-level structure
        assert root.tag == "oft"
        thincurr = root.find("thincurr")
        assert thincurr is not None

        # Resistivity entry
        eta_elem = thincurr.find("eta")
        assert eta_elem is not None
        eta_values = [float(v) for v in eta_elem.text.split(",")]
        assert np.allclose(eta_values, resistivity_list)

        # Two coil sets
        coil_sets = thincurr.findall("icoils/coil_set")
        assert len(coil_sets) == len([filament_1, filament_2])

        # First coil set has 2 points, second has 3
        coils = coil_sets[0].findall("coil")
        assert len(coils) == 1
        assert int(coils[0].attrib["npts"]) == 2

        coils2 = coil_sets[1].findall("coil")
        assert int(coils2[0].attrib["npts"]) == 3


def test_gen_OFT_sensors_file():
    """gen_OFT_sensors_file must produce a .loc file and return its path."""
    sensor_details = xr.Dataset(
        data_vars={
            "position": (
                ("sensor", "coord"),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ),
            "normal": (
                ("sensor", "coord"),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ),
            "radius": ("sensor", np.array([0.01, 0.01])),
        },
        coords={"sensor": ["A", "B"]},
        attrs={"sensor_set_name": "test"},
    )

    with tempfile.TemporaryDirectory() as working_directory:
        sensor_path = gen_OFT_sensors_file(sensor_details, working_directory)
        assert os.path.exists(sensor_path), "sensor file was not created"
        assert sensor_path.endswith(".loc")

        with open(sensor_path) as f:
            content = f.read()
        assert len(content) > 0


def test_calc_direct_response_matches_frequency_response(oft_env_fixture):
    """calc_direct_response must return the same direct component as calc_frequency_response."""

    major_radius = 1
    minor_radius_vessel = 0.35
    minor_radius_sensor = 0.34
    minor_radius_plasma = 0.3
    num_filaments = nextprime(16)
    mode = {"m": 2, "n": 1}

    sensor_details = xr.Dataset(
        data_vars={
            "position": (
                ("sensor", "coord"),
                np.array(
                    [
                        [major_radius + minor_radius_sensor, 0.0, 0.0],
                        [0.0, major_radius + minor_radius_sensor, 0.0],
                    ]
                ),
            ),
            "normal": (
                ("sensor", "coord"),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ),
            "radius": ("sensor", np.array([0.01, 0.01])),
        },
        coords={"sensor": ["sensor_x", "sensor_y"]},
        attrs={"sensor_set_name": "test"},
    )

    toroidal_tracer = ToroidalFilamentTracer(
        mode["m"],
        mode["n"],
        major_radius,
        0.0,
        minor_radius_plasma,
        base_num_points=32,
        scale_points=False,
        prevent_synthetic_structure=False,
    )

    with tempfile.TemporaryDirectory() as working_directory:
        torus_mesh = create_torus_mesh(
            major_radius, minor_radius_vessel, ntheta=16, nphi=64
        )
        torus_mesh_file = os.path.join(working_directory, "torus_mesh.h5")
        torus_mesh.write_to_file(torus_mesh_file)

        sensor_file_path = gen_OFT_sensors_file(sensor_details, working_directory)

        filament_list, current_list = toroidal_tracer.get_filament_list(
            num_filaments=num_filaments
        )
        gen_OFT_filament_and_eta_file(
            working_directory, filament_list, [1e-6] * len(filament_list)
        )

        total_response, direct_response_freq, vessel_response = calc_frequency_response(
            oft_env=oft_env_fixture,
            tracer=toroidal_tracer,
            freq=10e3,
            mesh_file=torus_mesh_file,
            working_directory=working_directory,
            sensor_file_path=sensor_file_path,
        )

        direct_response_only = calc_direct_response(
            oft_env=oft_env_fixture,
            tracer=toroidal_tracer,
            mesh_file=torus_mesh_file,
            sensor_details=sensor_details,
            sensor_file_path=sensor_file_path,
            working_directory=working_directory,
        )

    direct_from_freq = direct_response_freq
    direct_from_direct = np.array(
        [
            direct_response_only["direct_response_real"].values
            + 1j * direct_response_only["direct_response_imag"].values
        ]
    ).ravel()

    np.testing.assert_allclose(
        direct_from_direct,
        direct_from_freq,
        rtol=1e-6,
        err_msg="calc_direct_response must match the direct component of calc_frequency_response",
    )
