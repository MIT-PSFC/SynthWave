import os
import xarray as xr
import numpy as np
from synthwave import PACKAGE_ROOT
from synthwave.scratch.mesh_plot import create_torus_mesh
from synthwave.magnetic_geometry.filaments import ToroidalFilamentTracer
from synthwave.mirnov.prep_thincurr_input import (
    gen_OFT_filament_and_eta_file,
    gen_OFT_sensors_file,
)
from synthwave.mirnov.run_thincurr_model import calc_frequency_response

EXAMPLE_DIR = os.path.join(PACKAGE_ROOT, "..", "thincurr_scratch", "freq_response")

MAJOR_RADIUS = 1
MINOR_RADIUS = 0.5


if __name__ == "__main__":
    # Goal: Plot response of icoils on a toroidal mesh
    if not os.path.exists(EXAMPLE_DIR):
        os.makedirs(EXAMPLE_DIR)

    # Create torus mesh if it does not exist
    torus_mesh_file = os.path.join(EXAMPLE_DIR, "thincurr_torus_mesh.h5")
    if not os.path.exists(torus_mesh_file):
        torus_mesh = create_torus_mesh(MAJOR_RADIUS, MINOR_RADIUS)
        torus_mesh.write_to_file(torus_mesh_file)

    # Create 'oft_in.xml' file with icoil definitions
    toroidal_tracer = ToroidalFilamentTracer(2, 1, MAJOR_RADIUS, 0, MINOR_RADIUS)
    filament_list = toroidal_tracer.get_filament_list(num_filaments=10)

    oft_filament_file = os.path.join(EXAMPLE_DIR, "oft_in.xml")
    gen_OFT_filament_and_eta_file(
        working_directory=EXAMPLE_DIR,
        filament_list=filament_list,
        resistivity_list=[1e-6],
    )

    probe_details = xr.Dataset(
        data_vars={
            "position": (
                ("sensor", "coord"),
                np.array(
                    [
                        [MAJOR_RADIUS + 0.01, 0.0, 0.0],  # sensor on x axis
                        [0, MAJOR_RADIUS + 0.01, 0.0],  # sensor on y axis
                        [MAJOR_RADIUS, 0, MINOR_RADIUS + 0.01],  # sensor on z axis
                    ]
                ),
            ),
            "normal": (
                ("sensor", "coord"),
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
            ),
            "radius": ("sensor", np.array([0.01, 0.01, 0.01])),
        },
        coords={
            "sensor": np.array(["sensor_x", "sensor_y", "sensor_z"]),
        },
        attrs={
            "probe_set_name": "test_probes",
        },
    )
    gen_OFT_sensors_file(
        probe_details=probe_details,
        working_directory=EXAMPLE_DIR,
    )

    sensors_bode = calc_frequency_response(
        probe_details=probe_details,
        tracer=toroidal_tracer,
        freq=10e3,
        mesh_file=torus_mesh_file,
        working_directory=EXAMPLE_DIR,
    )
