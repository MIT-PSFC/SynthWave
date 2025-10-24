import os
import tempfile
from freeqdsk import geqdsk
import xarray as xr

from synthwave import PACKAGE_ROOT

from synthwave.mirnov.generate_synthetic_mirnov_signals import thincurr_synthetic_mirnov_signal

def test_thincurr_synthetic_mirnov_signal():
    mesh_file = os.path.join(PACKAGE_ROOT, "input_data", "cmod", "C_Mod_ThinCurr_VV-homology.h5")
    geqdsk_file = os.path.join(PACKAGE_ROOT, "input_data", "cmod", "g1051202011.1000")
    with open(geqdsk_file, 'r') as f:
        eqdsk = geqdsk.read(f)

    probe_details = xr.DataArray(
        data={
            "sensor_id": ["sensor_1", "sensor_2"],
            "x": [1.0, 0.5],
            "y": [0.0, -0.5],
            "z": [0.0, 0.2],
        }
    )

    # Resistivity of the conducting structure in Ohm-m
    eta = [1e-6]

    # Simulate a 2/1 mode at 10 kHz
    mode = {"m": 2, "n": 1, "m_pts": 100, "n_pts": 100}
    freq=10e3

    # Ignoring frequency response correction for now

    with tempfile.TemporaryDirectory() as working_directory:
        sensors_bode = thincurr_synthetic_mirnov_signal(
            probe_details=probe_details,
            mesh_model_file=mesh_file,
            eqdsk=eqdsk,
            freq=freq,
            mode=mode,
            eta=eta,
            working_directory=working_directory,
        )

        # Basic checks on the output
        assert sensors_bode is not None