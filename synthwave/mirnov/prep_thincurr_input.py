#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr

from typing import Optional


from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors


def gen_OFT_filament_and_eta_file(
    working_directory: str,
    filament_list: list[np.ndarray],
    resistivity_list: list[float],
    debug: Optional[bool] = False,
):
    """
    Write filament (x,y,z) coordinates to xml file in OFT format

    Args:
        working_directory (str): Directory to save the OFT file
        filament_coords (list[np.ndarray]): List of filaments, each filament is an array of shape (n_points, 3)
        resistivity_list (list[float]): List of resistivities for each conducting structure in Ohm-m
    """

    # Convert eta to string
    resistivity_str = ",".join([str(eta) for eta in resistivity_list])

    filament_file = os.path.join(working_directory, "oft_in.xml")

    # Write filaments and eta to file
    with open(filament_file, "w+") as f:
        f.write(f"<oft>\n\t<thincurr>\n\t<eta>{resistivity_str}</eta>\n\t<icoils>\n")

        if debug:
            print(f"File open for writing: {filament_file}")

        for filament in filament_list:
            # All coils in a coil set share the same current waveform,
            # So we group each filament into its own coil set
            f.write("\t<coil_set>\n")
            f.write(f'\n\t\t<coil npts="{np.shape(filament)[0]}" scale="1.0">\n')
            for xyz in filament:
                x = xyz[0]
                y = xyz[1]
                z = xyz[2]
                f.write("\t\t\t %1.3f, %1.3f, %1.3f\n" % (x, y, z))
            f.write("\t\t</coil>\n")
            f.write("\t</coil_set>\n")
        f.write("\t</icoils>\n\t</thincurr>\n</oft>")

    if debug:
        print("Wrote OFT filament file to %s" % filament_file)


def gen_OFT_sensors_file(
    sensor_details: xr.Dataset, working_directory: str, sensor_file_path: Optional[str] = None, debug: bool = True
):
    """
    Write sensor details to OFT format file for ThinCurr
    OFT calls them sensors, but the more common terminology is sensors.

    Args:
        sensor_details (xr.Dataset): Dataset containing sensor location and normal orientation in x,y,z
        working_directory (str): Directory to save the OFT sensor file
        debug (Optional[bool], default=True): If True, print debug information

    Returns:
        str: Path to the generated OFT sensor file
    """
    # Assume sensor_details is an xarray dataset with the following variables:
    # X, Y, Z (coordinates of each sensor)
    # theta, phi (orientation of each sensor)

    sensor_list = []
    for sensor in sensor_details.sensor.values:
        pt = sensor_details.position.sel(sensor=sensor).values
        # normal vector does not currently account for toroidal tilt
        norm = sensor_details.normal.sel(sensor=sensor).values
        # sensor radius
        dx = sensor_details.radius.sel(sensor=sensor).item()
        # create Mirnov object
        sensor_list.append(Mirnov(pt, norm, sensor, dx))

    if sensor_file_path is None:
        sensor_file_path = os.path.join(working_directory, f"floops_{sensor_details.attrs['sensor_set_name']}.loc")

    # Save in ThinCurr format
    save_sensors(
        sensor_list,
        sensor_file_path,
    )
    if debug:
        print(f"Wrote OFT sensor file to {sensor_file_path}")
    return sensor_file_path
