from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr
from scipy.constants import mu_0

from synthwave.magnetic_geometry.filaments import FilamentTracer

# OpenFUSIONToolkit loads its shared libraries on import, so the ThinCurr functions
# import it themselves and the Biot-Savart functions here stay usable without it
if TYPE_CHECKING:
    from OpenFUSIONToolkit import OFT_env


def filament_flux_matrix(
    sensor_details: xr.Dataset,
    filament_list: list,
    max_block_elements: int = 2**18,
) -> np.ndarray:
    """Vectorized Biot-Savart: unit-current flux of every filament through every sensor.

    Every polyline segment of every filament is one current element dl at its midpoint m.
    Its flux through a sensor at p with normal n is proportional to (dl x (p - m)) . n / |p - m|^3.
    The numerator expands to dl . (p x n) - n . (dl x m), so over all sensor and segment
    pairs it is one rank 6 matrix product. Only the distance is formed elementwise.
    The segments of all filaments are stacked and processed in blocks
    so the (n_sensors, n_segments) work arrays stay below max_block_elements each,
    which keeps them in cache.

    The direct response of any current pattern on these filaments is current_list @ flux,
    so modes sharing a filament set (harmonics on one rational surface) share this matrix.

    Args:
        sensor_details: Dataset with position (n_sensors, 3), normal (n_sensors, 3), radius (n_sensors,).
        filament_list: List of (N_pts, 3) cartesian filament point arrays. NaN points are dropped,
            a filament with fewer than 2 valid points contributes zero flux.
        max_block_elements: Upper bound on n_sensors * n_segments per block.

    Returns:
        Real array of shape (n_filaments, n_sensors), flux [Wb] per unit filament current [A].
    """
    sensor_positions = np.asarray(sensor_details["position"].data, dtype=float)
    sensor_normals = np.asarray(sensor_details["normal"].data, dtype=float)
    sensor_areas = np.pi * np.asarray(sensor_details["radius"].data, dtype=float) ** 2
    num_sensors = len(sensor_positions)

    dl_parts, midpoint_parts, owner_parts = [], [], []
    for i, filament_pts in enumerate(filament_list):
        filament_pts = np.asarray(filament_pts, dtype=float)
        filament_pts = filament_pts[~np.isnan(filament_pts).any(axis=1)]
        if len(filament_pts) < 2:
            continue
        dl_parts.append(filament_pts[1:] - filament_pts[:-1])
        midpoint_parts.append(0.5 * (filament_pts[1:] + filament_pts[:-1]))
        owner_parts.append(np.full(len(filament_pts) - 1, i))

    flux = np.zeros((len(filament_list), num_sensors))
    if not dl_parts:
        return flux
    dl = np.concatenate(dl_parts)
    midpoints = np.concatenate(midpoint_parts)
    owner = np.concatenate(owner_parts)

    # (dl x (p - m)) . n = dl . (p x n) - n . (dl x m)
    sensor_factors = np.hstack(
        (np.cross(sensor_positions, sensor_normals), -sensor_normals)
    )
    segment_factors = np.hstack((dl, np.cross(dl, midpoints)))

    block = max(1, max_block_elements // max(num_sensors, 1))
    for start in range(0, len(dl), block):
        stop = start + block
        # Distance components as (n_sensors, n_seg) arrays
        r_x = sensor_positions[:, 0:1] - midpoints[None, start:stop, 0]
        r_y = sensor_positions[:, 1:2] - midpoints[None, start:stop, 1]
        r_z = sensor_positions[:, 2:3] - midpoints[None, start:stop, 2]
        r2 = r_x * r_x + r_y * r_y + r_z * r_z
        contribution = sensor_factors @ segment_factors[start:stop].T
        contribution /= r2 * np.sqrt(r2)
        # Segments are stored filament by filament, so each filament in the block is one
        # contiguous run: reduce every run and add it to its owner row
        owner_block = owner[start:stop]
        run_starts = np.concatenate([[0], np.flatnonzero(np.diff(owner_block)) + 1])
        run_sums = np.add.reduceat(contribution, run_starts, axis=1)
        flux[owner_block[run_starts]] += run_sums.T
    return flux * (mu_0 / (4 * np.pi)) * sensor_areas[None, :]


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
    flux = filament_flux_matrix(sensor_details, filament_list)
    return np.asarray(current_list, dtype=complex) @ flux


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
    from OpenFUSIONToolkit.ThinCurr import ThinCurr

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
    from OpenFUSIONToolkit.ThinCurr import ThinCurr

    # working directory for caching inductance matrices, which can take a long time to compute
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
        import matplotlib.pyplot as plt
        import pyvista
        import vtk

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
