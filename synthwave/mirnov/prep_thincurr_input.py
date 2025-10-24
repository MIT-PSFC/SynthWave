#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction

from synthwave.magnetic_geometry.filaments import EquilibriumFilament, TraceType
from synthwave.magnetic_geometry.equilibrium_field import EquilibriumField

import matplotlib.pyplot as plt
from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors


################################################################################################
################################################################################################
def gen_coil_currs_sin_cos(mode, debug=True, doSave=False):
    # Assign currents to existing filaments for time dependent calculation
    # The ordering of the filaments and currents must match for the field topology to be correct

    m_pts = mode["m_pts"]
    m = mode["m"]
    n = mode["n"]
    n_pts = mode["n_pts"]

    # Need to get the list of starting angles, to correctly assign filament colors
    starting_angle, _ = gen_filament_coords(mode)  # returns as lists

    # # assume that m_pts can be a list
    # if type(m_pts) == int: num_filaments = m_pts*len(starting_angle)
    # else: num_filaments = sum([len(theta) for theta in starting_angle])
    if debug:
        print("Number of filaments: %d" % n_pts)

    # Coil currents for the sin/cos compoenents for each filement
    coil_currs = np.zeros((2, n_pts))

    # Time vector in first column
    if debug:
        print("Initialized coil currents array with shape (%d,%d)" % coil_currs.shape)

    for ind, phase in enumerate([0, np.pi / 2]):  # sin and cos components
        coil_currs[ind] = np.cos(n * starting_angle + phase)

    # Save coil currents to .npy for manual inspection later
    if doSave:
        np.savez(
            doSave + "coil_currs.npz",
            coil_currs=coil_currs,
            m=m,
            n=n,
            n_pts=n_pts,
            m_pts=m_pts,
        )

    return coil_currs


#########################################################################################################
def gen_filament_coords(params):
    # TODO(ZanderKeith): Ask Rian why you need to distribute along both poloidal and toroidal angles
    # Solely toroidal should be enough, yes?
    m = params["m"]
    n = params["n"]
    n_pts = params["n_pts"]
    m_pts = params["m_pts"]
    # generate theta,phi coordinates for filaments
    # The points launch in a fractional sector of the toroidal plane
    # and wrap enough times to return to their starting point
    # Ported from geqdsk_filament_generator.py
    # Only accepts single m/n pair for now

    ratio = Fraction(m, n)
    m_local = ratio.numerator
    n_local = ratio.denominator

    starting_angle = np.linspace(0, 2 * np.pi / n_local, n_pts, endpoint=False)
    winding_angle = np.linspace(0, 2 * np.pi * m_local, m_pts)

    return starting_angle, winding_angle


#########################################################################################################
##########################################################################################
#########################################################################
#########################################################################
# Field Line Following Model
##########################################################################
#########################################################################


#########################################################################
# Initial phi positions for filaments to launch from
def starting_phi(m, n, m_pts, n_pts):
    ratio = Fraction(m, n)
    m_local = ratio.numerator
    n_local = ratio.denominator
    phi_start = np.linspace(0, 2 * np.pi / n_local, n_pts, endpoint=False)
    phi_advance = np.linspace(0, 2 * np.pi * m_local, m_pts)
    return phi_start, phi_advance


###############################################################
# Core of new Winding Method
def wind_in_theta(eqdsk, m, n, debug=False):
    eq_field = EquilibriumField(eqdsk)
    eq_filament = EquilibriumFilament(m, n, eq_field)
    filament_points, filament_etas = eq_filament._trace(
        method=TraceType.AVERAGE, num_filament_points=300
    )

    if debug:
        plt.close("Winding_Theta")
        plt.figure(num="Winding_Theta", tight_layout=True)
        plt.plot(filament_points[:, 1] / (2 * np.pi))
        plt.plot(
            [0, len(filament_points)], [0, filament_points[-1, 1] / (2 * np.pi)], "--k"
        )
        plt.grid()
        plt.ylabel(r"$\hat{\phi}$ [2$\pi$ Rad]")
        plt.xlabel("Filament Points")
        plt.show()

    # Nan Check
    return filament_points, filament_etas


#######################################
def conv_theta_wind_to_coords(filament_points, phi_start):
    # Convert R, Phi, Z for each filament, with starting launch offset in phi
    coords = []

    for phi_0 in phi_start:
        phi_ = filament_points[:, 1] + phi_0  # local phi for a given filament
        X = filament_points[:, 0] * np.cos(phi_)
        Y = filament_points[:, 0] * np.sin(phi_)
        Z = filament_points[:, 2]
        coords.append([X, Y, Z])

    return np.array(coords)


def calc_filament_coords_field_lines(mode, geqdsk, debug=False):
    # Calling container for field-tracing filaments
    # Ported from geqdsk_filament_generator.py

    m = mode["m"]
    n = mode["n"]
    n_pts = mode["n_pts"]
    m_pts = mode["m_pts"]
    # generate theta,phi coordinates for filaments
    # The points launch in a fractional sector of the poloidal plane, and
    # wrap toroidally enough times to return to their starting point

    phi_start, phi_advance = starting_phi(m, n, m_pts, n_pts)

    filament_points, filament_etas = wind_in_theta(geqdsk, m, n, debug=debug)

    coords = conv_theta_wind_to_coords(filament_points, phi_start)

    return coords


################################################################################################
################################################################################################
# Generate OFT Input Files
################################################################################################
def gen_OFT_filement_and_eta_file(filament_file, filament_coords, eta, debug=False):
    """ """
    # Write filament (x,y,z) coordinates to xml file in OFT format
    # Each filament is a list of points, each point is a list of [x,y,z] points
    # Removing options for multiple m,n modes for now
    # eta is the resistivity of the conducting structure in Ohm-m

    # Convert eta to string
    eta = ",".join([str(e) for e in eta])

    # Write filaments and eta to file
    with open(filament_file + "oft_in.xml", "w+") as f:
        f.write("<oft>\n\t<thincurr>\n\t<eta>%s</eta>\n\t<icoils>\n" % eta)

        if debug:
            print("File open for writing: %s" % filament_file)

        for filament in filament_coords:
            f.write("\t<coil_set>\n")
            f.write('\n\t\t<coil npts="%d" scale="1.0">\n' % np.shape(filament)[1])

            for xyz in np.array(filament).T:
                x = xyz[0]
                y = xyz[1]
                z = xyz[2]
                f.write("\t\t\t %1.3f, %1.3f, %1.3f\n" % (x, y, z))

            f.write("\t\t</coil>\n")
            f.write("\t</coil_set>\n")

        f.write("\t</icoils>\n\t</thincurr>\n</oft>")

    if debug:
        print("Wrote OFT filament file to %s" % filament_file)


################################################################
#######3#########################################################
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
