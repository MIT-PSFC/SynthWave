#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

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
            f.write("\t<coil_set>\n")
            f.write(f'\n\t\t<coil npts="{np.shape(filament)[0]}" scale="1.0">\n')

            for xyz in filament:
                (
                    x,
                    y,
                    z,
                ) = xyz
                f.write("\t\t\t %1.3f, %1.3f, %1.3f\n" % (x, y, z))

            f.write("\t\t</coil>\n")
            f.write("\t</coil_set>\n")

        f.write("\t</icoils>\n\t</thincurr>\n</oft>")

    if debug:
        print("Wrote OFT filament file to %s" % filament_file)


def gen_OFT_sensors_file(probe_details, working_directory, debug=True):
    # Assume probe_details is an xarray dataset with the following variables:
    # X, Y, Z (coordinates of each probe)
    # theta, phi (orientation of each probe)

    sensor_list = []
    for probe in probe_details.sensor.values:
        pt = probe_details.position.sel(sensor=probe).values
        # normal vector does not current account for toroidal tilt
        norm = probe_details.normal.sel(sensor=probe).values
        # Probe radius
        dx = probe_details.radius.sel(sensor=probe).item()
        # create Mirnov object
        sensor_list.append(Mirnov(pt, norm, probe, dx))

    # Save in ThinCurr format
    save_sensors(
        sensor_list,
        f"{working_directory}/floops_{probe_details.attrs['probe_set_name']}.loc",
    )
    if debug:
        print(
            "Wrote OFT sensor file to %s/floops_%s.loc"
            % (working_directory, probe_details.attrs["probe_set_name"])
        )
    return sensor_list
