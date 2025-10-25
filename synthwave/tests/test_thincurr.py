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

@pytest.mark.parametrize("mode", [{"m": 2, "n": 1}, {"m": 3, "n": 2}, {"m": 3, "n": 1}])
def test_toroidal_angles(mode):
    major_radius = 1
    minor_radius_vessel = 0.35
    minor_radius_probe = 0.34
    minor_radius_plasma = 0.3

    num_filaments = 30
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

        toroidal_tracer = ToroidalFilamentTracer(mode["m"], mode["n"], major_radius, 0, minor_radius_plasma)
        filament_list = toroidal_tracer.get_filament_list(num_filaments=num_filaments)

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

    # Phase difference between sensors should roughly match cylindrical approximation
    toroidal_phase_ab, toroidal_phase_cd = np.pi/2, np.pi/2
    expected_phase_diff_ab = toroidal_phase_ab * mode["n"]
    expected_phase_diff_cd = toroidal_phase_cd * mode["n"]
    measured_phase_diff_ab = np.abs(total_response_phase[1] - total_response_phase[0])
    measured_phase_diff_cd = np.abs(total_response_phase[3] - total_response_phase[2])

    poloidal_phase_ac, poloidal_phase_bd = np.pi/2, np.pi/2
    expected_phase_diff_ac = poloidal_phase_ac * mode["m"]
    expected_phase_diff_bd = poloidal_phase_bd * mode["m"]
    measured_phase_diff_ac = np.abs(total_response_phase[2] - total_response_phase[0])
    measured_phase_diff_bd = np.abs(total_response_phase[3] - total_response_phase[1])

    assert np.isclose(measured_phase_diff_ab, expected_phase_diff_ab, atol=0.001)
    assert np.isclose(measured_phase_diff_cd, expected_phase_diff_cd, atol=0.001)
    assert np.isclose(measured_phase_diff_ac, expected_phase_diff_ac, atol=0.001)
    assert np.isclose(measured_phase_diff_bd, expected_phase_diff_bd, atol=0.001)
