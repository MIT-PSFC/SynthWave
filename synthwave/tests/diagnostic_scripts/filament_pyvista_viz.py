# PyVista filament and sensor visualization for mode phases
from diagnostic_header import (
    EquilibriumField,
    EquilibriumFilamentTracer,
    convert_cocos,
    np,
    os,
    pyvista,
    xr,
)
from tars.config import config
from tars.out_of_scope.chisq_fast.sensor_helpers import (
    build_sensor_details_compat,
    normalize_eq_field_dataset,
)
from tars.reconstruct.utils import build_geqdsk


def build_equilibrium_field(ds_eq_time: xr.Dataset) -> EquilibriumField:
    eq_data = normalize_eq_field_dataset(ds_eq_time)
    geqdsk = build_geqdsk(eq_data)
    # cocos = detect_cocos(geqdsk) # Cocos check for diagnostic purposes
    geqdsk_converted = convert_cocos(geqdsk, cocos_target=1)
    return EquilibriumField(geqdsk_converted)


def get_filaments_and_currents(mode, eq_field, config):
    tracer = EquilibriumFilamentTracer(
        mode[0],
        mode[1],
        eq_field=eq_field,
        base_num_points=config.direct_response_filament_points,
        default_trace_type=EquilibriumFilamentTracer.TraceType.FIELD,
    )
    filament_list, current_list = tracer.get_filament_list(
        num_filaments=60,  # config.direct_response_filaments,
        coordinate_system="cartesian",
    )
    return filament_list, np.array(current_list, dtype=complex)


def get_sensor_positions(ds_eq_time: xr.Dataset):
    sensor_details = build_sensor_details_compat(
        ds_eq_time, sensor_set_name="mirnov_custom"
    )
    positions = sensor_details["position"].data
    names = (
        sensor_details["sensor_name"].data if "sensor_name" in sensor_details else None
    )
    return positions, names


def plot_filament_phase_comparison(
    mode,
    filament_list,
    current_list,
    sensor_positions,
    sensor_names,
    save_path,
    doublePlot=False,
    showFilaments=False,
):
    pyvista.OFF_SCREEN = False

    phase_labels = (
        ["Cosine phase", "Sine phase"]
        if doublePlot
        else ["Sensor Locations for ModeSpec"]
    )
    phase_values = [0.0, np.pi / 2] if doublePlot else [0]

    plotter = pyvista.Plotter(
        shape=(1, 2 if doublePlot else 1),
        off_screen=False,
        window_size=(1600 if doublePlot else 800, 400),
    )

    cmap = "coolwarm"
    current_scalar_name = "phase_current"

    for index, (label, phase) in enumerate(zip(phase_labels, phase_values)):
        plotter.subplot(0, index)
        plotter.add_text(label, font_size=14, position="upper_left")
        if showFilaments:
            for filament, current in zip(filament_list, current_list):
                pts = np.asarray(filament, dtype=float)
                values = np.full(pts.shape[0], np.real(current * np.exp(1j * phase)))
                spline = pyvista.Spline(pts, len(pts))
                spline[current_scalar_name] = values
                tube = spline.tube(radius=0.005 * 4)
                plotter.add_mesh(
                    tube,
                    scalars=current_scalar_name,
                    cmap=cmap,
                    scalar_bar_args={"title": "Current [A]", "vertical": True},
                    opacity=1.0,
                )

    inds = [
        ind
        for ind in range(len(sensor_names))
        if ("MPI66" in sensor_names[ind] or "322" in sensor_names[ind])
    ]
    plotter.add_points(
        sensor_positions[inds],
        color="black",
        point_size=20,
        render_points_as_spheres=True,
        label="Sensors",
    )
    if sensor_names is not None:
        plotter.add_point_labels(
            sensor_positions[inds],
            [str(name) for name in sensor_names[inds]],
            font_size=10,
            point_size=20,
            text_color="black",
            shape_opacity=0.0,
            always_visible=True,
        )

    plotter.view_isometric()
    plotter.show_axes()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plotter.link_views()
    # plotter.screenshot(save_path)
    plotter.camera.elevation -= 5
    plotter.camera.azimuth = 45
    pos = plotter.camera.position
    # plotter.camera.zoom('tight')
    plotter.camera.position = np.array(pos) * 0.65

    plotter.save_graphic(save_path)
    plotter.show()

    plotter.close()
    print(f"Saved filament phase comparison to {save_path}")


##############################################################################################


def plot_modes_filaments(ds_shot_path, eq_time_s, modes, output_dir, save_ext=""):
    ds_shot = xr.open_dataset(ds_shot_path)
    eq_time_idx = np.argmin(np.abs(ds_shot.time.values - eq_time_s))
    ds_eq_time = ds_shot.sel(time_idx=eq_time_idx)
    eq_field = build_equilibrium_field(ds_eq_time)
    sensor_positions, sensor_names = get_sensor_positions(ds_eq_time)

    for mode in modes if np.ndim(modes) == 2 else [modes]:
        filament_list, current_list = get_filaments_and_currents(mode, eq_field, config)
        save_path = os.path.join(
            output_dir,
            f"mode_m{mode[0]}_n{mode[1]}_filament_phase_comparison_{ds_shot_path.split('/')[-1].split('.')[0]}{save_ext}.pdf",
        )
        plot_filament_phase_comparison(
            mode, filament_list, current_list, sensor_positions, sensor_names, save_path
        )


if __name__ == "__main__":
    # ds_shot_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/179118.nc"
    # eq_time_idx = 23000
    # ds_shot_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/1120906030.nc"
    # eq_time_idx = 1200#23000
    # ds_shot_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/174956.nc"
    ds_shot_path = (
        "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/tars_input_180516.nc"
    )
    eq_time_s = 2.5  # 23000
    modes = [(5, 1)]
    output_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "output_plots", "filament_pyvista"
        )
    )
    save_ext = "_DIII_D_Sensors_Only"
    plot_modes_filaments(ds_shot_path, eq_time_s, modes, output_dir, save_ext)
