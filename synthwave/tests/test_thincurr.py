import os
import numpy as np
import tempfile
from freeqdsk import geqdsk
import xarray as xr
import pytest

from synthwave import PACKAGE_ROOT

from synthwave.mirnov.prep_thincurr_input import gen_OFT_filament_and_eta_file
from synthwave.scratch.mesh_plot import create_torus_mesh
from synthwave.mirnov.generate_synthetic_mirnov_signals import thincurr_synthetic_mirnov_signal
from synthwave.magnetic_geometry.filaments import ToroidalFilamentTracer
from synthwave.mirnov.run_thincurr_model import calc_frequency_response
from synthwave.mirnov.prep_thincurr_input import gen_OFT_sensors_file, gen_OFT_filament_and_eta_file
from synthwave.magnetic_geometry.utils import phi_domain

@pytest.mark.parametrize("mode", [{"m": 2, "n": 1}, {"m": 3, "n": 2}, {"m": -3, "n": 2}, {"m": 3, "n": 1}])
@pytest.mark.parametrize("major_radius", [1, 20])
def test_toroidal_angles(mode, major_radius):
    minor_radius_vessel = 0.35
    minor_radius_probe = 0.34
    minor_radius_plasma = 0.3

    num_filaments = 12
    resistivity = 1e-6
    num_filament_points = 100 * major_radius
    probe_details = xr.Dataset(
        data_vars={
            "position": (
                ("sensor", "coord"),
                np.array(
                    [
                        [major_radius + minor_radius_probe, 0.0, 0.0],  # sensor on x axis
                        [0, major_radius + minor_radius_probe, 0.0],    # sensor on y axis
                        [major_radius, 0, minor_radius_probe],          # sensor up top on x axis
                        [0, major_radius, minor_radius_probe],          # sensor up top on y axis
                    ]
                ),
            ),
            "normal": (
                ("sensor", "coord"),
                np.array(
                    [
                        # Normals pointing radially inward
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                    ]
                ),
            ),
            "radius": ("sensor", np.array([0.01, 0.01, 0.01, 0.01])),
        },
        coords={
            "sensor": np.array(["sensor_a", "sensor_b", "sensor_c", "sensor_d"]),
        },
        attrs={
            "probe_set_name": "test_probes",
        },
    )


    with tempfile.TemporaryDirectory() as working_directory:
        torus_mesh_file = os.path.join(working_directory, "thincurr_torus_mesh.h5")
        torus_mesh = create_torus_mesh(major_radius, minor_radius_vessel)
        torus_mesh.write_to_file(torus_mesh_file)

        toroidal_tracer = ToroidalFilamentTracer(mode["m"], mode["n"], major_radius, 0, minor_radius_plasma, num_points=num_filament_points)
        filament_list = toroidal_tracer.get_filament_list(num_filaments=num_filaments, coordinate_system="cartesian")

        gen_OFT_filament_and_eta_file(
            working_directory=working_directory,
            filament_list=filament_list,
            resistivity_list=[resistivity],
        )

        gen_OFT_sensors_file(
            probe_details=probe_details,
            working_directory=working_directory,
        )

        total_response, direct_response, vessel_response = calc_frequency_response(
            probe_details=probe_details,
            tracer=toroidal_tracer,
            freq=10e3,
            mesh_file=torus_mesh_file,
            working_directory=working_directory,
        )

    total_response_phase = np.angle(total_response)
    direct_response_phase = np.angle(direct_response)
    vessel_response_phase = np.angle(vessel_response)

    # Phase difference for the direct response should closely match cylindrical approximation
    toroidal_phase_ab, toroidal_phase_cd = np.pi/2, np.pi/2
    expected_phase_diff_ab = toroidal_phase_ab * mode["n"]
    expected_phase_diff_cd = toroidal_phase_cd * mode["n"]
    direct_measured_phase_diff_ab = direct_response_phase[1] - direct_response_phase[0]
    direct_measured_phase_diff_cd = direct_response_phase[3] - direct_response_phase[2]

    poloidal_phase_ac, poloidal_phase_bd = np.pi/2, np.pi/2
    expected_phase_diff_ac = poloidal_phase_ac * mode["m"]
    expected_phase_diff_bd = poloidal_phase_bd * mode["m"]
    direct_measured_phase_diff_ac = direct_response_phase[2] - direct_response_phase[0]
    direct_measured_phase_diff_bd = direct_response_phase[3] - direct_response_phase[1]

    assert np.isclose(direct_measured_phase_diff_ab, expected_phase_diff_ab, atol=0.001)
    assert np.isclose(direct_measured_phase_diff_cd, expected_phase_diff_cd, atol=0.001)
    if major_radius == 1:
        # At small major radius, toroidal approximation is less accurate
        # In this case just make sure the two values match closely since they should be identical
        assert np.isclose(direct_measured_phase_diff_ac, direct_measured_phase_diff_bd, atol=0.001)
    else:
        # At large major radius, toroidal approximation should still be fairly accurate
        assert np.isclose(direct_measured_phase_diff_ac, expected_phase_diff_ac, atol=0.01)
        assert np.isclose(direct_measured_phase_diff_bd, expected_phase_diff_bd, atol=0.01)

    # Even with vessel response, toroidal phase difference should closely match cylindrical approximation
    # This is because vessel response should also be axisymmetric
    total_measured_phase_diff_ab = total_response_phase[1] - total_response_phase[0]
    total_measured_phase_diff_cd = total_response_phase[3] - total_response_phase[2]
    assert np.isclose(total_measured_phase_diff_ab, expected_phase_diff_ab, atol=0.001)
    assert np.isclose(total_measured_phase_diff_cd, expected_phase_diff_cd, atol=0.001)

    # Poloidal phase difference should deviate from cylindrical approximation due to vessel effects
    total_measured_phase_diff_ac = total_response_phase[2] - total_response_phase[0]
    total_measured_phase_diff_bd = total_response_phase[3] - total_response_phase[1]
    assert not np.isclose(total_measured_phase_diff_ac, expected_phase_diff_ac, atol=0.1)
    assert not np.isclose(total_measured_phase_diff_bd, expected_phase_diff_bd, atol=0.1)


def test_thincurr_input():
    mode = {"m": 2, "n": 1}
    major_radius = 1
    minor_radius_vessel = 0.35
    minor_radius_probe = 0.34
    minor_radius_plasma = 0.3

    num_filaments = 12
    resistivity = 1e-6

    probe_details = xr.Dataset(
        data_vars={
            "position": (
                ("sensor", "coord"),
                np.array(
                    [
                        [major_radius + minor_radius_probe, 0.0, 0.0],  # sensor on x axis
                        [0, major_radius + minor_radius_probe, 0.0],    # sensor on y axis
                        [major_radius, 0, minor_radius_probe],          # sensor up top on x axis
                        [0, major_radius, minor_radius_probe],          # sensor up top on y axis
                    ]
                ),
            ),
            "normal": (
                ("sensor", "coord"),
                np.array(
                    [
                        # Normals pointing radially inward
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                    ]
                ),
            ),
            "radius": ("sensor", np.array([0.01, 0.01, 0.01, 0.01])),
        },
        coords={
            "sensor": np.array(["sensor_a", "sensor_b", "sensor_c", "sensor_d"]),
        },
        attrs={
            "probe_set_name": "test_probes",
        },
    )


    with tempfile.TemporaryDirectory() as working_directory:
        torus_mesh_file = os.path.join(working_directory, "thincurr_torus_mesh.h5")
        torus_mesh = create_torus_mesh(major_radius, minor_radius_vessel)
        torus_mesh.write_to_file(torus_mesh_file)

        toroidal_tracer = ToroidalFilamentTracer(mode["m"], mode["n"], major_radius, 0, minor_radius_plasma, num_points=100)
        filament_list = toroidal_tracer.get_filament_list(num_filaments=num_filaments, coordinate_system="cartesian")

        gen_OFT_filament_and_eta_file(
            working_directory=working_directory,
            filament_list=filament_list,
            resistivity_list=[resistivity],
            debug=True,
        )

        gen_OFT_sensors_file(
            probe_details=probe_details,
            working_directory=working_directory,
            debug=True,
        )