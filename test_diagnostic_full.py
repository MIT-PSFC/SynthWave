#!/usr/bin/env python3
"""Full diagnostic test with all assertions"""

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
    print("Starting full diagnostic test...")

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
    oft_env = get_oft_env_singleton()
    print("✓ OFT_env created")

    with tempfile.TemporaryDirectory() as working_directory:
        print(f"Working directory: {working_directory}")

        torus_mesh_file = os.path.join(working_directory, "thincurr_torus_mesh.h5")
        torus_mesh = create_torus_mesh(major_radius, minor_radius_vessel)
        torus_mesh.write_to_file(torus_mesh_file)
        print("✓ Mesh created")

        toroidal_tracer = ToroidalFilamentTracer(
            mode["m"], mode["n"], major_radius, 0, minor_radius_plasma,
            base_num_points=base_num_points,
            scale_points=False,
            prevent_synthetic_structure=True,
        )
        print("✓ Tracer created")

        filament_list, _ = toroidal_tracer.get_filament_list(
            num_filaments=num_filaments, coordinate_system="cartesian"
        )
        print(f"✓ Got {len(filament_list)} filaments")

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
        print("✓ Sensor file generated")

        print("\nCalculating frequency response...")
        total_response, direct_response, vessel_response = calc_frequency_response(
            oft_env=oft_env,
            tracer=toroidal_tracer,
            freq=10e3,
            mesh_file=torus_mesh_file,
            working_directory=working_directory,
            sensor_file_path=sensor_file_path,
        )
        print("✓ Response calculated")

        print("\n" + "=" * 70)
        print("RUNNING ALL TEST ASSERTIONS")
        print("=" * 70)

        total_response_phase = np.angle(total_response)
        direct_response_phase = np.angle(direct_response)
        vessel_response_phase = np.angle(vessel_response)

        # Phase difference for the direct response should closely match cylindrical approximation
        toroidal_phase_ab, toroidal_phase_cd = np.pi / 2, np.pi / 2
        expected_phase_diff_ab = angle_domain(toroidal_phase_ab * mode["n"])
        expected_phase_diff_cd = angle_domain(toroidal_phase_cd * mode["n"])
        direct_measured_phase_diff_ab = np.angle(direct_response[1] / direct_response[0])
        direct_measured_phase_diff_cd = np.angle(direct_response[3] / direct_response[2])

        poloidal_phase_ac, poloidal_phase_bd = np.pi / 2, np.pi / 2
        expected_phase_diff_ac = angle_domain(poloidal_phase_ac * mode["m"])
        expected_phase_diff_bd = angle_domain(poloidal_phase_bd * mode["m"])
        direct_measured_phase_diff_ac = np.angle(direct_response[2] / direct_response[0])
        direct_measured_phase_diff_bd = np.angle(direct_response[3] / direct_response[1])

        print("\n1. DIRECT RESPONSE - TOROIDAL PHASE DIFFERENCES")
        print(f"   Expected phase diff AB: {expected_phase_diff_ab:.6f}")
        print(f"   Measured phase diff AB: {direct_measured_phase_diff_ab:.6f}")
        diff_ab = wrapped_diff(direct_measured_phase_diff_ab, expected_phase_diff_ab)
        print(f"   Wrapped diff: {diff_ab:.6f} (tolerance: 0.01)")
        try:
            assert np.isclose(diff_ab, 0, atol=0.01)
            print("   ✓ PASS")
        except AssertionError:
            print("   ✗ FAIL")

        print(f"\n   Expected phase diff CD: {expected_phase_diff_cd:.6f}")
        print(f"   Measured phase diff CD: {direct_measured_phase_diff_cd:.6f}")
        diff_cd = wrapped_diff(direct_measured_phase_diff_cd, expected_phase_diff_cd)
        print(f"   Wrapped diff: {diff_cd:.6f} (tolerance: 0.01)")
        try:
            assert np.isclose(diff_cd, 0, atol=0.01)
            print("   ✓ PASS")
        except AssertionError:
            print("   ✗ FAIL")

        print("\n2. DIRECT RESPONSE - POLOIDAL PHASE DIFFERENCES (major_radius=1)")
        print(f"   Expected phase diff AC: {expected_phase_diff_ac:.6f}")
        print(f"   Measured phase diff AC: {direct_measured_phase_diff_ac:.6f}")
        print(f"   Measured phase diff BD: {direct_measured_phase_diff_bd:.6f}")
        diff_ac_bd = wrapped_diff(direct_measured_phase_diff_ac, direct_measured_phase_diff_bd)
        print(f"   Wrapped diff AC-BD: {diff_ac_bd:.6f} (tolerance: 0.01)")
        try:
            assert np.isclose(diff_ac_bd, 0, atol=0.01)
            print("   ✓ PASS")
        except AssertionError:
            print("   ✗ FAIL")

        print("\n3. TOTAL RESPONSE - TOROIDAL PHASE DIFFERENCES")
        total_measured_phase_diff_ab = np.angle(total_response[1] / total_response[0])
        total_measured_phase_diff_cd = np.angle(total_response[3] / total_response[2])
        print(f"   Expected phase diff AB: {expected_phase_diff_ab:.6f}")
        print(f"   Measured phase diff AB: {total_measured_phase_diff_ab:.6f}")
        total_diff_ab = wrapped_diff(total_measured_phase_diff_ab, expected_phase_diff_ab)
        print(f"   Wrapped diff: {total_diff_ab:.6f} (tolerance: 0.001)")
        try:
            assert np.isclose(total_diff_ab, 0, atol=0.001)
            print("   ✓ PASS")
        except AssertionError:
            print(f"   ✗ FAIL - diff {total_diff_ab:.6f} exceeds tolerance 0.001")

        print(f"\n   Expected phase diff CD: {expected_phase_diff_cd:.6f}")
        print(f"   Measured phase diff CD: {total_measured_phase_diff_cd:.6f}")
        total_diff_cd = wrapped_diff(total_measured_phase_diff_cd, expected_phase_diff_cd)
        print(f"   Wrapped diff: {total_diff_cd:.6f} (tolerance: 0.001)")
        try:
            assert np.isclose(total_diff_cd, 0, atol=0.001)
            print("   ✓ PASS")
        except AssertionError:
            print(f"   ✗ FAIL - diff {total_diff_cd:.6f} exceeds tolerance 0.001")

        print("\n4. TOTAL RESPONSE - POLOIDAL PHASE DIFFERENCES")
        total_measured_phase_diff_ac = np.angle(total_response[2] / total_response[0])
        total_measured_phase_diff_bd = np.angle(total_response[3] / total_response[1])
        print(f"   Expected phase diff AC: {expected_phase_diff_ac:.6f}")
        print(f"   Measured phase diff AC: {total_measured_phase_diff_ac:.6f}")
        total_diff_ac_expected = wrapped_diff(total_measured_phase_diff_ac, expected_phase_diff_ac)
        print(f"   Wrapped diff from expected: {total_diff_ac_expected:.6f} (should NOT be close to 0)")
        try:
            assert not np.isclose(total_diff_ac_expected, 0, atol=0.1)
            print("   ✓ PASS (correctly different from cylindrical)")
        except AssertionError:
            print(f"   ✗ FAIL - should differ from expected by >0.1, but diff is {total_diff_ac_expected:.6f}")

        print(f"\n   Expected phase diff BD: {expected_phase_diff_bd:.6f}")
        print(f"   Measured phase diff BD: {total_measured_phase_diff_bd:.6f}")
        total_diff_bd_expected = wrapped_diff(total_measured_phase_diff_bd, expected_phase_diff_bd)
        print(f"   Wrapped diff from expected: {total_diff_bd_expected:.6f} (should NOT be close to 0)")
        try:
            assert not np.isclose(total_diff_bd_expected, 0, atol=0.1)
            print("   ✓ PASS (correctly different from cylindrical)")
        except AssertionError:
            print(f"   ✗ FAIL - should differ from expected by >0.1, but diff is {total_diff_bd_expected:.6f}")

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
