#!/usr/bin/env python3
"""Minimal test to diagnose remaining issues"""

import os
import sys
import tempfile
import numpy as np
import xarray as xr
from sympy import nextprime
from OpenFUSIONToolkit import OFT_env

from synthwave.scratch.mesh_plot import create_torus_mesh
from synthwave.magnetic_geometry.filaments import ToroidalFilamentTracer
from synthwave.mirnov.prep_thincurr_input import gen_OFT_filament_and_eta_file, gen_OFT_sensors_file
from synthwave.mirnov.run_thincurr_model import calc_frequency_response
from synthwave.magnetic_geometry.utils import angle_domain, wrapped_diff


_OFT_ENV_SINGLETON = None


def get_oft_env_singleton():
    """Return a single OFT_env instance per Python process."""
    global _OFT_ENV_SINGLETON
    if _OFT_ENV_SINGLETON is None:
        _OFT_ENV_SINGLETON = OFT_env()
    return _OFT_ENV_SINGLETON

def main():
    print("Starting diagnostic test...")

    # Simple test case
    mode = {"m": 2, "n": 1}
    major_radius = 1
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
                np.array([
                    [major_radius + minor_radius_sensor, 0.0, 0.0],
                    [0, major_radius + minor_radius_sensor, 0.0],
                    [major_radius, 0, minor_radius_sensor],
                    [0, major_radius, minor_radius_sensor],
                ]),
            ),
            "normal": (
                ("sensor", "coord"),
                np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]),
            ),
            "radius": ("sensor", np.array([0.01, 0.01, 0.01, 0.01])),
        },
        coords={"sensor": np.array(["sensor_a", "sensor_b", "sensor_c", "sensor_d"])},
        attrs={"sensor_set_name": "test_sensors"},
    )

    print("\nCreating OFT_env...")
    try:
        oft_env = get_oft_env_singleton()
        print("✓ OFT_env created successfully")
    except Exception as e:
        print(f"✗ Failed to create OFT_env: {e}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as working_directory:
        print(f"\nWorking directory: {working_directory}")

        print("\n1. Creating mesh...")
        try:
            torus_mesh_file = os.path.join(working_directory, "thincurr_torus_mesh.h5")
            torus_mesh = create_torus_mesh(major_radius, minor_radius_vessel)
            torus_mesh.write_to_file(torus_mesh_file)
            print(f"✓ Mesh created: {torus_mesh_file}")
        except Exception as e:
            print(f"✗ Failed to create mesh: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\n2. Creating tracer...")
        try:
            toroidal_tracer = ToroidalFilamentTracer(
                mode["m"], mode["n"], major_radius, 0, minor_radius_plasma,
                base_num_points=base_num_points,
                scale_points=False,
                prevent_synthetic_structure=True,
            )
            print("✓ Tracer created")
        except Exception as e:
            print(f"✗ Failed to create tracer: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\n3. Getting filament list...")
        try:
            filament_list, _ = toroidal_tracer.get_filament_list(
                num_filaments=num_filaments, coordinate_system="cartesian"
            )
            print(f"✓ Got {len(filament_list)} filaments")
        except Exception as e:
            print(f"✗ Failed to get filament list: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\n4. Generating OFT files...")
        try:
            gen_OFT_filament_and_eta_file(
                working_directory=working_directory,
                filament_list=filament_list,
                resistivity_list=[resistivity],
            )
            print("✓ Filament file generated")

            sensor_file_path = gen_OFT_sensors_file(
                sensor_details=sensor_details,
                working_directory=working_directory,
            )
            print(f"✓ Sensor file generated: {sensor_file_path}")
        except Exception as e:
            print(f"✗ Failed to generate OFT files: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\n5. Calling calc_frequency_response...")
        try:
            total_response, direct_response, vessel_response = calc_frequency_response(
                oft_env=oft_env,
                tracer=toroidal_tracer,
                freq=10e3,
                mesh_file=torus_mesh_file,
                working_directory=working_directory,
                sensor_file_path=sensor_file_path,
            )
            print(f"✓ Response calculated")
            print(f"  Total response: {total_response}")
            print(f"  Direct response: {direct_response}")
            print(f"  Vessel response: {vessel_response}")
        except Exception as e:
            print(f"✗ Failed to calculate response: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\n6. Running assertions from test...")
        try:
            total_response_phase = np.angle(total_response)
            direct_response_phase = np.angle(direct_response)
            vessel_response_phase = np.angle(vessel_response)

            toroidal_phase_ab, toroidal_phase_cd = np.pi / 2, np.pi / 2
            expected_phase_diff_ab = angle_domain(toroidal_phase_ab * mode["n"])
            expected_phase_diff_cd = angle_domain(toroidal_phase_cd * mode["n"])
            direct_measured_phase_diff_ab = np.angle(direct_response[1] / direct_response[0])
            direct_measured_phase_diff_cd = np.angle(direct_response[3] / direct_response[2])

            print(f"  Expected phase diff AB: {expected_phase_diff_ab}")
            print(f"  Measured phase diff AB: {direct_measured_phase_diff_ab}")
            print(f"  Wrapped diff AB: {wrapped_diff(direct_measured_phase_diff_ab, expected_phase_diff_ab)}")

            assert np.isclose(
                wrapped_diff(direct_measured_phase_diff_ab, expected_phase_diff_ab),
                0,
                atol=0.01,
            ), "Phase diff AB assertion failed"
            print("✓ Phase diff AB assertion passed")

            assert np.isclose(
                wrapped_diff(direct_measured_phase_diff_cd, expected_phase_diff_cd),
                0,
                atol=0.01,
            ), "Phase diff CD assertion failed"
            print("✓ Phase diff CD assertion passed")

            print("\n✓ ALL ASSERTIONS PASSED!")

        except AssertionError as e:
            print(f"✗ Assertion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error during assertions: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\nDiagnostic test completed successfully!")


if __name__ == "__main__":
    main()
