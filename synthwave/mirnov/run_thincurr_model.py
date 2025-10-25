from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr import ThinCurr
import numpy as np
import xarray as xr
import pyvista
import os
import matplotlib.pyplot as plt
from synthwave.magnetic_geometry.filaments import FilamentTracer
from typing import Optional


def calc_frequency_response(
    probe_details: xr.Dataset,
    tracer: FilamentTracer,
    freq: float,
    mesh_file: str,
    working_directory: str,
    n_threads: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the measured frequency response at the given probes due to filaments defined by the tracer.
    Assumes that the OFT input files have already been generated in the working_directory.

    Returns:
        total_response (np.ndarray): Complex array of total probe signals [T]
        vessel_response (np.ndarray): Complex array of probe signals due to vessel currents [T]
        direct_response (np.ndarray): Complex array of probe signals due to direct filament coupling [T]

    """

    # Create thin wall model
    oft_env = OFT_env(nthreads=os.cpu_count() if n_threads is None else n_threads)
    tw_model = ThinCurr(oft_env)
    tw_model.setup_model(
        mesh_file=mesh_file,
        xml_filename=os.path.join(working_directory, "oft_in.xml"),
    )
    tw_model.setup_io(working_directory)

    # Calculate mutual inductances

    # finite element mesh -> sensor, coil -> sensor
    probe_set_file = os.path.join(
        working_directory, f"floops_{probe_details.attrs['probe_set_name']}.loc"
    )
    Msensor, Msc, sensor_obj = tw_model.compute_Msensor(probe_set_file)

    # filament -> finite element mesh
    Mc = tw_model.compute_Mcoil()

    # Build inductance matrix because apparently that's required
    tw_model.compute_Lmat(
        use_hodlr=True,
        cache_file=os.path.join(
            working_directory,
            f"HOLDR_L_{os.path.basename(mesh_file)}_{probe_details.attrs['probe_set_name']}.save",
        ),
    )
    tw_model.compute_Rmat()

    # Build driver from filaments
    filament_details = tracer.make_points_and_currents(
        num_filaments=Mc.shape[0], coordinate_system="cartesian"
    )
    filament_currents = (
        filament_details.current.values
    )  # Complex array for rotating wave

    driver = np.zeros((2, tw_model.nelems))
    # Driver represents the complex phasor: real and imaginary parts
    driver[0, :] = np.dot(filament_currents.real, Mc)
    driver[1, :] = np.dot(filament_currents.imag, Mc)

    # Calculate mesh response at given frequency
    mesh_response = tw_model.compute_freq_response(fdriver=driver, freq=freq)

    # Contribution from mesh current to the sensor
    vessel_response_matrix = np.dot(mesh_response, Msensor)
    vessel_response = vessel_response_matrix[0, :] + 1j * vessel_response_matrix[1, :]

    # Contribution from filament current directly to the sensor
    # This is the mutual inductance flux: Phi = M * I (both are complex)
    direct_response = np.dot(filament_currents, Msc)

    total_response = vessel_response + direct_response

    return total_response, vessel_response, direct_response


################################################################################################
################################################################################################
def get_mesh(mesh_file, working_directory, n_threads, sensor_set, debug=True):
    # Load mesh, compute inductance matrices

    if debug:
        print("Loading mesh from %s" % mesh_file)
    oft_env = OFT_env(nthreads=os.cpu_count() if n_threads == 0 else n_threads)
    tw_mesh = ThinCurr(oft_env)
    tw_mesh.setup_model(
        mesh_file=working_directory + mesh_file,
        xml_filename=working_directory + "oft_in.xml",
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
    working_directory,
    coil_current_magnitude=1,
):
    Mcoil = tw_mesh.compute_Mcoil()
    driver = np.zeros((2, tw_mesh.nelems))
    driver[:, :] = np.dot(coil_currs, Mcoil)

    # Mutual between the mesh and sensors, and coil and sensors
    Msensor, Msc, _ = tw_mesh.compute_Msensor(
        working_directory + "floops_%s.loc" % probe_details.attrs["probe_set_name"]
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
    working_directory="",
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
        print("Plotted filaments")

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
        p.save_graphic(working_directory + "Mesh_and_Filaments%s.pdf" % save_Ext)
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
    working_directory,
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
            working_directory
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
                    working_directory
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
