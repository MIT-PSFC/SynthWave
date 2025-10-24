"""
Main script to generate synthetic mirnov signals using ThinCurr
Requires a valid conducting structure mesh for ThinCurr
and an equilibrium file in gEQDSK format
and a set of probe locations and orientations
and a mode to simulate (m,n)
and a resistivity for the conducting structure
and a frequency to simulate

Outputs a netcdf file with the real and imaginary components of the probe signals in [T/s]
and optional plots of the mesh, filaments, and sensors

Original by Rian Chandra,
Modified by Zander Keith
"""

import os
import numpy as np
import xarray as xr
from typing import Optional
from freeqdsk import geqdsk
from synthwave import PACKAGE_ROOT

from synthwave.magnetic_geometry.filaments import BaseFilament

from synthwave.mirnov.prep_thincurr_input import (
    gen_coil_currs_sin_cos,
    gen_filament_coords,
    calc_filament_coords_field_lines,
    gen_OFT_sensors_file,
    gen_OFT_filement_and_eta_file,
)

from synthwave.mirnov.run_thincurr_model import (
    get_mesh,
    run_frequency_scan,
    makePlots,
    correct_frequency_response,
)


def synthetic_mirnov_signal(
    probe_details: xr.Dataset,
    filament: BaseFilament,
    freq: float,
    working_directory: str,
    mesh_file: Optional[str] = None,
    vessel_resistivity: Optional[float] = 1e-6,
    debug: Optional[bool] = False,
) -> xr.Dataset:
    """
    Calculate the real and imaginary components of the magnetic probe signals using ThinCurr.
    If a mesh_file is not provided, does not include the conducting structure effects.

    Args:
        probe_details (xr.Dataset): Dataset containing probe geometry in X,Y,Z coordinate, orientation in theta, phi,
        filament (BaseFilament): Filament object defining the mode to simulate
        freq (float): Frequency of the mode to simulate
        working_directory (str): Directory to write and load ThinCurr files
        mesh_file (Optional[str]): Path to the vessel model file for ThinCurr
        vessel_resistivity (Optional[float], default=1e-6): Resistivity of the conducting structure in Ohm-m
        debug (Optional[bool], default=False): If True, print debug information

    Returns:
        xr.Dataset: Dataset containing the simulated probe signals in [T/s].
    """

    # Prepare filament currents and put them in OFT format
    # https://openfusiontoolkit.github.io/OpenFUSIONToolkit/docs/v1.0.0-beta6/doc_tw_main.html#doc_tw_main_filament

    # Prepare sensors in OFT format

    # Prepare ThinCurr Model, get finite element Mesh

    # Calculate frequency response


################################################################################################
################################################################################################
def thincurr_synthetic_mirnov_signal(
    probe_details: xr.Dataset,
    mesh_model_file: str,
    eqdsk: geqdsk,
    freq: float,
    mode: dict,
    eta: list[float],
    working_directory: str,
    save_Ext: str = "",
    debug: bool = False,
    n_threads: int = 0,
    sensor_freq_response: dict = {},
    doPlot: bool = False,
    doSave: bool = False,
    plotParams: dict = {"clim_J": [0, 0.5]},
) -> xr.Dataset:
    """
    Calculate the real and imaginary components of the magnetic probe signals using ThinCurr.
    Assumes that a valid conducting structure mesh is provided.


    Args:
        probe_details: Dataset containing probe geometry in X,Y,Z coordinate, orientation in theta, phi,
        mesh_model: Path to the vessel model file for ThinCurr
        eqdsk: Equilibrium field data from a gEQDSK file
        freq: Frequency of the mode to simulate
        mode: dictionary containing m, n and m_pts and n_pts to set the filament resolution
        debug: If True, print debug information
        n_threads: Number of threads to use for ThinCurr calculations. If 0, use all available threads.
        sensor_freq_response: Dictionary with sensor names as keys and frequency response functions as values,
            if empty, assume that the sensors have a flat response
        doPlot: If True, generate debug plots of the mesh, filaments, and sensors
        doSave: If True, save the output dataset to a netcdf file
        plotParams: Dictionary with plotting parameters, e.g. {'clim_J': [0, 0.5]} for color limits of eddy current plot
        working_directory: Directory to store and load mesh and temporary files from
    Returns:
        xr.Dataset: Dataset containing the simulated probe signals in [T/s].
    """

    ######################################################################################
    # Prepare filament currents, locations

    # Generate coil currents (for artificial mode)
    coil_currs = gen_coil_currs_sin_cos(mode)

    # Get starting coorindates for filaments
    theta, phi = gen_filament_coords(mode)

    filament_coords = calc_filament_coords_field_lines(mode, eqdsk, debug=debug)

    # Put filaments in OFT file format
    gen_OFT_filement_and_eta_file(working_directory, filament_coords, eta)

    ######################################################################################
    # Generate sensors in OFT format
    sensors = gen_OFT_sensors_file(probe_details, working_directory, debug=debug)

    ######################################################################################
    # Prepare ThinCurr Model, Get finite element Mesh
    tw_mesh, sensor_obj, Mc = get_mesh(
        mesh_model_file, working_directory, n_threads, sensor_set, debug=debug
    )

    ######################################################################################
    # Calculate frequency response
    sensors_bode = run_frequency_scan(
        tw_mesh,
        freq,
        coil_currs,
        probe_details,
        mesh_model_file,
        sensor_obj,
        mode,
        working_directory,
    )

    # Correct for sensor frequency response and save results
    correct_frequency_response(
        sensors_bode,
        sensor_freq_response,
        freq,
        mode,
        doSave,
        debug,
        working_directory,
        probe_details,
        save_Ext,
    )

    ######################################################################################
    # Optional debug output
    # Plot mesh, filaments, sensors, and currents
    makePlots(
        tw_mesh,
        mode,
        coil_currs,
        sensors,
        doSave,
        save_Ext,
        filament_coords,
        plot_B_surf=True,
        debug=debug,
        plotParams=plotParams,
        doPlot=doPlot,
        working_directory=working_directory,
    )

    ######################################################################################

    return sensors_bode


###############################################################################
###############################################################################
###############################################################################
# Example code to generate a response in the C-Mod mesh
if __name__ == "__main__":
    sensor_set = "C_MOD_ALL"
    # Load example probe details
    probe_details_file = os.path.join(
        PACKAGE_ROOT,
        "input_data",
        "cmod",
        f"sensor_details_{sensor_set}.nc",
    )
    probe_details = xr.load_dataset(probe_details_file)

    # Load example equilibrium
    file_geqdsk = "g1051202011.1000"

    # Define mode to simulate
    mode = {"m": 4, "n": 2, "m_pts": 20, "n_pts": 40}

    # Resistivity of the conducting structure in Ohm-m
    eta = [1e-6]

    # Path to the mesh model file for ThinCurr
    mesh_model = "C_Mod_ThinCurr_VV-homology.h5"

    # Frequency to simulate
    freq = 30000  # Hz

    #####################
    # Frequency response correction [ leave as a flat response if this is unknown: H = lambda w: 1 ]
    R = 6
    L = 60e-6
    C = 760e-12
    R_0 = 0.7

    def Z_R(w):
        out = R_0 * np.ones_like(w)
        # Final factor is sqrt[resistivity/permeability]
        out[w / (2 * np.pi) >= 220e3] *= (
            1
            + (np.sqrt(w[w / (2 * np.pi) >= 220e3] / (2 * np.pi)) - np.sqrt(220e3))
            * 0.0043
        )  # Skin depth correction
        out += R  #
        return out

    def Z_L(w):
        return 1j * w * L

    def Z_C(w):
        return 1 / (1j * w * C)

    def Z_total(w):
        return Z_R(w) + Z_L(w) + Z_C(w)

    def H(w):
        return Z_C(w) / Z_total(w)  # Voltage across capacitor / input voltage

    # Frequency response dictionary for each sensor
    freq_response_dict = {name: H for name in probe_details.coords["sensor"].values}

    ###################
    debug = True
    working_directory = "input_data/"
    save_Ext = "_CMod"
    n_threads = 12
    doSave = True
    doPlot = True
    plotParams = {"clim_J": [0, 0.5]}

    # Run ThinCurr sensor response simulation
    sensors_bode = thincurr_synthetic_mirnov_signal(
        probe_details,
        mesh_model,
        file_geqdsk,
        freq,
        mode,
        eta,
        working_directory=working_directory,
        save_Ext=save_Ext,
        debug=debug,
        n_threads=n_threads,
        sensor_freq_response=freq_response_dict,
        doSave=doSave,
        doPlot=doPlot,
        plotParams=plotParams,
    )

    # Print results
    if debug:
        print(
            f"Simulated frequency: {freq / 1e3} kHz, mode (m,n)=({mode['m']},{mode['n']})"
        )
        print("Probe signals [T/s]:")
        for i, sensor_name in enumerate(sensors_bode.sensor.values):
            print(
                f"Sensor: {sensor_name}, Real: {sensors_bode.sel(sensor=sensor_name).signal.values.real:.2e},"
                + f" Imag: {sensors_bode.sel(sensor=sensor_name).signal.values.imag:.2e}"
            )
