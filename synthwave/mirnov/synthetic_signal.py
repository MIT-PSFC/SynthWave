import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyvista
import vtk
import xarray as xr
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from scipy.constants import mu_0

from synthwave.magnetic_geometry.filaments import FilamentTracer

def direct_response_biot_savart(
    sensor_details: xr.Dataset,
    filament_list: list,
    current_list: list,
) -> np.ndarray:
    """Vectorized Biot-Savart direct response: filaments -> sensor flux.

    Args:
        sensor_details: Dataset with position (n_sensors, 3), normal (n_sensors, 3), radius (n_sensors,).
        filament_list: List of (N_pts, 3) filament point arrays.
        current_list: List of complex currents, one per filament.

    Returns:
        Complex array of shape (n_sensors,): total flux through each sensor.
    """
    sensor_positions = sensor_details["position"].data  # (n_sensors, 3)
    sensor_normals = sensor_details["normal"].data  # (n_sensors, 3)
    sensor_areas = np.pi * sensor_details["radius"].data ** 2  # (n_sensors,)

    direct_response = np.zeros(len(sensor_positions), dtype=complex)
    for filament_pts, current in zip(filament_list, current_list):
        filament_pts = np.asarray(filament_pts, dtype=float)
        valid = ~np.isnan(filament_pts).any(axis=1)
        filament_pts = filament_pts[valid]
        if len(filament_pts) < 2:
            continue
        # r_prime: (n_sensors, n_pts, 3)
        r_prime = sensor_positions[:, None, :] - filament_pts[None, :, :]
        r_prime_norm = np.linalg.norm(r_prime, axis=2)  # (n_sensors, n_pts)
        dl = np.gradient(filament_pts, axis=0)  # (n_pts, 3)
        dl_cross_r = np.cross(dl[None, :, :], r_prime)  # (n_sensors, n_pts, 3)
        B_total = np.sum(
            (mu_0 / (4 * np.pi)) * dl_cross_r / r_prime_norm[:, :, None] ** 3,
            axis=1,
        )  # (n_sensors, 3)
        flux = np.sum(B_total * sensor_normals, axis=1) * sensor_areas  # (n_sensors,)
        direct_response += current * flux
    return direct_response

def direct_response_thincurr(
    oft_env: OFT_env,
    tracer: FilamentTracer,
    mesh_file: str,
    sensor_details: xr.Dataset,
    sensor_file_path: str,
    working_directory: str,
) -> xr.Dataset:
    """
    Calculate only the direct filament to sensor responses (no vessel currents) at the given sensors due to filaments defined by the tracer.

    From testing, the vessel response has minimal impact on the phases, so this can be used for spectral analysis.
    """

    # Create thin wall model
    tw_model = ThinCurr(oft_env)
    try:
        tw_model.setup_model(
            mesh_file=mesh_file,
            xml_filename=os.path.join(working_directory, "oft_in.xml"),
        )

        tw_model.setup_io(working_directory)
    except Exception as e:
        print(f"Error setting up ThinCurr model: {e}")
        print(f"Mesh file: {mesh_file}")
        print(f"xml file: {os.path.join(working_directory, 'oft_in.xml')}")
        raise e

    # Calculate mutual inductances

    # finite element mesh -> sensor, coil -> sensor
    # This should be fast since we're using a simple vessel mesh
    _, Msc, sensor_obj = tw_model.compute_Msensor(sensor_file_path)

    # Build driver from filaments
    filament_details = tracer.get_filament_ds(
        num_filaments=Msc.shape[0], coordinate_system="cartesian"
    )
    filament_currents = (
        filament_details.current.values
    )  # Complex array for rotating wave

    # Contribution from filament current directly to the sensor
    # This is the mutual inductance flux: Phi = M * I (both are complex)
    direct_response = np.dot(filament_currents, Msc)

    direct_response_ds = xr.Dataset(
        data_vars={
            "direct_response_real": (
                ["sensor_idx"],
                direct_response.real,
            ),
            "direct_response_imag": (
                ["sensor_idx"],
                direct_response.imag,
            ),
        },
        coords={
            "sensor_idx": sensor_obj[
                "names"
            ]  # Define the 'sensor_idx' coordinate with the list of sensor names
        },
        attrs={
            "mesh_file": mesh_file,
            "sensor_set_name": sensor_details.attrs["sensor_set_name"],
        },
    )

    return direct_response_ds


def frequency_response_thincurr(
    oft_env: OFT_env,
    tracer: FilamentTracer,
    freq: float,
    mesh_file: str,
    working_directory: str,
    sensor_file_path: Optional[str] = None,
    sensor_details: Optional[xr.Dataset] = None,
    debug_plot_path: Optional[str] = None,
    vessel_cache_path: Optional[str] = None,
    msensor_cache_path: Optional[str] = None,
    mcoil_cache_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the measured frequency response at the given sensors due to filaments defined by the tracer.
    Assumes that the OFT input files have already been generated in the working_directory.

    Also note that this isn't exactly what a sensor would measure, since the output is in [T] not [T/s].
    This is fine for mode structure identification, but for amplitude matching the output needs to be corrected elsewhere.

    Args:
        oft_env (OFT_env): OFT environment object, can only be created once per process
        tracer (FilamentTracer): FilamentTracer object defining the filaments to simulate
        freq (float): Frequency to simulate [Hz]
        mesh_file (str): Path to the vessel mesh file for ThinCurr
        working_directory (str): Directory to read/write ThinCurr files
        sensor_file_path (str): Path to the sensor file for ThinCurr
        sensor_details (xr.Dataset): Dataset containing details about the sensors, used for debug plotting
        debug_plot_path (str, optional): If provided, path prefix to save debug plots


    Returns:
        total_response (np.ndarray): Complex array of total sensor signals [T]
        direct_response (np.ndarray): Complex array of sensor signals due to direct filament coupling [T]
        vessel_response (np.ndarray): Complex array of sensor signals due to vessel currents [T]
    """

    # Directory for caching inductance matrices, which can take a long time to compute
    # ThinCurr checks the hashes of input files to determine if cache is valid

    # Create thin wall model
    tw_model = ThinCurr(oft_env)
    tw_model.setup_model(
        mesh_file=mesh_file,
        xml_filename=os.path.join(working_directory, "oft_in.xml"),
    )
    tw_model.setup_io(working_directory)

    # Calculate mutual inductances

    # finite element mesh -> sensor, coil -> sensor
    Msensor, Msc, _sensor_obj = tw_model.compute_Msensor(
        sensor_file=sensor_file_path,
        cache_file=msensor_cache_path,
    )

    # filament -> finite element mesh
    Mc = tw_model.compute_Mcoil(
        cache_file=mcoil_cache_path,
    )

    # Build inductance matrix
    tw_model.compute_Lmat(
        cache_file=vessel_cache_path,
        use_hodlr=True,
    )
    tw_model.compute_Rmat()

    # Build driver from filaments
    filament_details = tracer.get_filament_ds(
        num_filaments=Mc.shape[0], coordinate_system="cartesian"
    )
    filament_currents = (
        filament_details.current.values
    )  # Complex array for rotating wave

    # Driver represents the complex phasor: real and imaginary parts
    driver = np.zeros((2, tw_model.nelems))
    driver[0, :] = np.dot(filament_currents.real, Mc)
    driver[1, :] = np.dot(filament_currents.imag, Mc)

    # Calculate mesh response at given frequency
    mesh_response_matrix = tw_model.compute_freq_response(fdriver=driver, freq=freq)

    # Contribution from mesh current to the sensor
    vessel_response_matrix = np.dot(mesh_response_matrix, Msensor)
    vessel_response = vessel_response_matrix[0, :] + 1j * vessel_response_matrix[1, :]

    # Contribution from filament current directly to the sensor
    # This is the mutual inductance flux: Phi = M * I (both are complex)
    direct_response = np.dot(filament_currents, Msc)

    total_response = direct_response + vessel_response

    if debug_plot_path is not None:
        # Only plotting to file, don't try to use a display
        pyvista.OFF_SCREEN = True
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
        tw_model.save_current(mesh_response_matrix[0, :], "Jr_coil")
        tw_model.save_current(mesh_response_matrix[1, :], "Ji_coil")
        plot_data = tw_model.build_XDMF()

        grid = plot_data["ThinCurr"]["smesh"].get_pyvista_grid()
        Jfull = plot_data["ThinCurr"]["smesh"].get_field("Jr_coil")

        pyvista.global_theme.allow_empty_mesh = True
        plotter = pyvista.Plotter()

        # Plot vessel mesh with eddy currents
        plotter.add_mesh(
            grid,
            color=[0, 0, 0, 0],
            use_transparency=True,
            opacity=0.8,
            show_edges=True,
            scalars=Jfull,
            clim=[0, np.max(np.abs(Jfull))],
            smooth_shading=True,
            scalar_bar_args={"title": "Eddy Current [A/m]"},
        )

        # Save vessel mesh plot
        plotter.screenshot(f"{debug_plot_path}_vessel.png", transparent_background=True)

        # Plot some filaments
        plot_filaments, plot_currents = tracer.get_filament_list(num_filaments=20)
        for filament, current in zip(plot_filaments, plot_currents):
            filament_spline = pyvista.Spline(filament, len(filament))

            plotter.add_mesh(
                filament_spline,
                color=plt.get_cmap("plasma")(
                    (current.real / np.max(np.array(plot_currents).real) + 1) / 2
                ),
                line_width=6,
                render_points_as_spheres=True,
                opacity=1,
            )

        # sensor_details = sensor_obj["details"]  # Hack for now

        # Plot sensors
        for sensor in sensor_details.sensor_idx:
            sensor_data = sensor_details.sel(sensor_idx=sensor)
            sensor_point = sensor_data.position.data
            plotter.add_points(
                sensor_point,
                color="k",
                point_size=10,
                render_points_as_spheres=True,
            )
        plotter.screenshot(
            f"{debug_plot_path}_filaments.png", transparent_background=True
        )

        # Have the view be top-down
        plotter.view_xy()

        plotter.render()

        plotter.screenshot(
            f"{debug_plot_path}_topdown.png", transparent_background=True
        )

        # Have the view be a slice through the xz plane
        plotter.camera_position = [
            (0, -0.1, 0),  # Position of the camera, set a little back on y-axis
            (0, 0, 0),  # Focal point at the origin
            (0, 0, 1),  # View up direction along z-axis
        ]

        plotter.render()

        plotter.screenshot(
            f"{debug_plot_path}_xzplane.png", transparent_background=True
        )

        # # Have the view be a slice through the xz plane
        # plotter.camera_position = [
        #     (0, -0.1, 0),  # Position of the camera, set a little back on y-axis
        #     (.8, 0, .5),  # Focal point at the mag-ax
        #     (1, 0, 0),  # View up direction along x-axis
        # ]
        plotter.view_zy()
        plotter.render()

        plotter.screenshot(
            f"{debug_plot_path}_xyplane.png", transparent_background=True
        )

        plotter.close()

    return total_response, direct_response, vessel_response