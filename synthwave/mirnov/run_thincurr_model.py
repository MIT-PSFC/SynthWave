#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr import ThinCurr
import numpy as np
import xarray as xr
import pyvista
import os
import matplotlib.pyplot as plt


################################################################################################
################################################################################################
def get_mesh(mesh_file, working_files_directory, n_threads, sensor_set, debug=True):
    # Load mesh, compute inductance matrices

    if debug:
        print("Loading mesh from %s" % mesh_file)
    oft_env = OFT_env(nthreads=os.cpu_count() if n_threads == 0 else n_threads)
    tw_mesh = ThinCurr(oft_env)
    tw_mesh.setup_model(
        mesh_file=working_files_directory + mesh_file,
        xml_filename=working_files_directory + "oft_in.xml",
    )
    tw_mesh.setup_io()

    # Sensor - mesh and sensor - filament inductances
    if debug:
        print(("Computing mutual inductances between mesh and %s sensors" % sensor_set))
    Msensor, Msc, sensor_obj = tw_mesh.compute_Msensor(
        "input_data/floops_%s.loc" % sensor_set
    )

    # Filament - mesh inductance
    if debug:
        print("Computing mutual inductances between mesh and filaments")
    Mc = tw_mesh.compute_Mcoil()

    # Build inductance matrix
    if debug:
        print("Computing mesh self-inductance")
    tw_mesh.compute_Lmat(
        use_hodlr=True,
        cache_file="input_data/HOLDR_L_%s_%s.save" % (mesh_file, sensor_set),
    )

    # Buld resistivity matrix
    if debug:
        print("Computing mesh resistivity matrix")
    tw_mesh.compute_Rmat()

    return tw_mesh, sensor_obj, Mc


################################################################################################
################################################################################################
def run_frequency_scan(
    tw_mesh,
    freq,
    coil_currs,
    probe_details,
    mesh_file,
    sensor_obj,
    mode,
    working_files_directory,
    coil_current_magnitude=1,
):
    Mcoil = tw_mesh.compute_Mcoil()
    driver = np.zeros((2, tw_mesh.nelems))
    driver[:, :] = np.dot(coil_currs, Mcoil)

    # Mutual between the mesh and sensors, and coil and sensors
    Msensor, Msc, _ = tw_mesh.compute_Msensor(
        working_files_directory
        + "floops_%s.loc" % probe_details.attrs["probe_set_name"]
    )

    # Test one frequency
    result = tw_mesh.compute_freq_response(fdriver=driver, freq=freq)

    # contribution from the mesh current to the sensor, with the mesh current at a given frequency
    probe_signals = np.dot(result, Msensor)

    # Contribuio from the coil current directly to the sensor
    probe_signals[:, :] += np.dot(coil_currs, Msc)

    # for i in range(probe_signals.shape[1]):
    #     print('Real: {0:13.5E}, Imaginary: {1:13.5E}'.format(*probe_signals[:,i]))
    probe_signals = (
        probe_signals[0, :] + 1j * probe_signals[1, :]
    )  # Combine real and imaginary parts

    # Only compute the mesh induced currents once
    tw_mesh.save_current(result[0, :], "Jr_coil")
    tw_mesh.save_current(result[1, :], "Ji_coil")
    _ = tw_mesh.build_XDMF()

    # Convert to xarray
    sensors_body = xr.Dataset(
        data_vars={
            "signal": (
                ["sensor"],
                probe_signals,
            )  # Use a single data variable with a 'sensor' dimension
        },
        coords={
            "sensor": sensor_obj[
                "names"
            ]  # Define the 'sensor' coordinate with the list of sensor names
        },
        attrs={
            "mesh_file": mesh_file,
            "sensor_set_name": probe_details.attrs["probe_set_name"],
            "driving_frequency": freq,
            "m": mode["m"],
            "n": mode["n"],
        },
    )

    return sensors_body


################################################################################################
################################################################################################
def makePlots(
    tw_mesh,
    mode,
    coil_currs,
    sensors,
    doSave,
    save_Ext,
    filament_coords,
    plot_B_surf=True,
    debug=True,
    plotParams={"clim_J": [0, 1]},
    doPlot=True,
    working_files_directory="",
):
    # Generate plots of mesh, filaments, sensors, and currents
    # Will plot induced current on the mesh if plot_B_surf is True

    if not doPlot:
        return []

    if debug:
        print("Generating output plots")
    # Mesh and Filaments
    m = mode["m"]
    n = mode["n"]

    # New ThinCurr way of getting mesh
    plot_data = tw_mesh.build_XDMF()
    grid = plot_data["ThinCurr"]["smesh"].get_pyvista_grid()
    if plot_B_surf:
        Jfull = plot_data["ThinCurr"]["smesh"].get_field("Jr_coil")
    if debug:
        print("Built Pyvista grid from ThinCurr mesh")

    pyvista.global_theme.allow_empty_mesh = True
    p = pyvista.Plotter()
    if debug:
        print("Launched Plotter")

    # Plot Mesh
    if plot_B_surf:
        p.add_mesh(
            grid,
            color="white",
            opacity=0.6,
            show_edges=True,
            scalars=Jfull,
            clim=plotParams["clim_J"],
            smooth_shading=True,
            scalar_bar_args={"title": "Eddy Current [A/m]"},
        )
    else:
        p.add_mesh(grid, color="white", opacity=0.9, show_edges=True)

    tmp = []
    if debug:
        print("Plotted Mesh")

    ###############################################
    # Plot Filaments
    # Modify to accept filament_coords as a list

    for ind, filament in enumerate(filament_coords):
        pts = np.array(filament).T
        p.add_points(
            pts[0],
            render_points_as_spheres=True,
            opacity=1,
            point_size=20,
            color="k",
            label="Launch Point" if ind == 0 else None,
        )

        spl = pyvista.Spline(pts, len(pts))

        p.add_mesh(
            spl,
            color=plt.get_cmap("plasma")(
                (coil_currs[0, ind] / np.max(coil_currs[0, :]) + 1) / 2
            ),
            line_width=10,
            render_points_as_spheres=True,
            label="Filament %d/%d" % (m, n) if ind == 0 else None,
            opacity=1,
        )

        tmp.append(pts)
    if debug:
        print("Plotted Fillaments")

    ###################################################
    # Plot Sensors
    for ind, s in enumerate(sensors):
        p.add_points(
            np.mean(s._pts, axis=0),
            color="k",
            point_size=10,
            render_points_as_spheres=True,
            label="Mirnov" if ind == 0 else None,
        )
    p.add_legend()
    if debug:
        print("Plotted Sensors")
    if doSave:
        p.save_graphic(working_files_directory + "Mesh_and_Filaments%s.pdf" % save_Ext)
    if debug:
        print("Saved figure")
    p.show()
    if debug:
        print("Plotted Figure")

    plt.show()
    return []


########################
########################################################################
def correct_frequency_response(
    sensors_bode,
    sensor_freq_response,
    freq,
    mode,
    doSave,
    debug,
    working_files_directory,
    probe_details,
    save_Ext,
):
    # Correct sensor signals for frequency response
    # Assume that the sensor sensor_freq_response is a dictionary with keys as sensor names
    # and values as a lambda function that takes frequency as input and returs the complex correction factor
    # e.g. sensor_correction = {'Mirnov1': lambda f: 1/(1+1j*f/1000), 'Mirnov2': lambda f: 1/(1+1j*f/2000)}
    # also correct into Bdot by multiplying by 2*pi*f

    for i, sensor_name in enumerate(sensors_bode["sensor"].values):
        sensors_bode.loc[{"sensor": sensor_name}] *= sensor_freq_response[sensor_name](
            np.array([freq])
        )[0] * (2 * np.pi * freq)

    if doSave:
        sensors_bode.to_netcdf(
            working_files_directory
            + "probe_signals_%s_m%02d_n%02d_f%1.1ekHz%s.nc"
            % (
                probe_details.attrs["probe_set_name"],
                mode["m"],
                mode["n"],
                freq / 1e3,
                save_Ext,
            ),
            auto_complex=True,
        )
        if debug:
            print(
                "Saved probe signals to %s"
                % (
                    working_files_directory
                    + "probe_signals_%s_m%02d_n%02d_f%1.1ekHz%s.nc"
                    % (
                        probe_details.attrs["probe_set_name"],
                        mode["m"],
                        mode["n"],
                        freq / 1e3,
                        save_Ext,
                    )
                )
            )

################################################################################################
################################################################################################
def plot_sensor_output(
        working_files_directory,
        probe_details,
        mode,
        freq,
        save_Ext,
        doSave,
):

    # Load in saved sensor signals from netCDF format and plot

    fName = working_files_directory + \
        "probe_signals_%s_m%02d_n%02d_f%1.1ekHz%s.nc" % \
        (
            probe_details.attrs["probe_set_name"], 
            mode["m"],
            mode["n"],
            freq / 1e3,
            save_Ext,
        )

    # Load data
    sensors_bode = xr.load_dataset(fName)

    # Generate signal magnitudes
    mags = [ np.linalg.norm(sensors_bode.sel(sensor=sig).signal.values.tolist()) \
            for sig in sensors_bode.sensor.values]
    sensor_names = sensors_bode.sensor.values.tolist()

    # Plot
    plt.close('Sensor_Signals_m%02d_n%02d_f%1.1ekHz'%( mode["m"],mode["n"],freq/1e3))
    fig,ax = plt.subplots(1,1,figsize=(6,5),tight_layout=True,num='Sensor_Signals_m%02d_n%02d_f%1.1ekHz'%(
        mode["m"],mode["n"],freq/1e3))
    sc = ax.plot(sensor_names, mags, '-*',label='Sensor Signals Magnitude')
    
    ax.set_xticks(range(0, len(sensor_names), 2))
    ax.set_xticklabels([sensor_names[i] for i in range(0, len(sensor_names), 2)])
    ax.tick_params(axis='x', rotation=90)
    # ax.legend(fontsize=8,handlelength=1)
    ax.set_ylabel('Signal Magnitude [T/s]')

    ax.grid()

    if doSave:
        fig.savefig(working_files_directory + 
            "Sensor_Signals_m%02d_n%02d_f%1.1ekHz%s.pdf" % 
            (
                mode["m"],
                mode["n"],
                freq / 1e3,
                save_Ext,
            ), transparent=True)
