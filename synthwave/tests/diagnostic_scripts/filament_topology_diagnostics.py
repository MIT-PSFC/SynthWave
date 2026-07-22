# Diagnostic plots for filaments and currents
# Purpose: allow for simpler debugging of tracer outputs
from diagnostic_header import (
    EquilibriumField,
    EquilibriumFilamentTracer,
    build_geqdsk,
    build_sensor_details_compat,
    config,
    convert_cocos,
    direct_response_biot_savart,
    normalize_eq_field_dataset,
    np,
    plt,
    xr,
)


def plot_filament_geometry(ds_shot_path, eq_time_idx, modes, doSave):
    # Get magnetic equilibrium unindent does not match any outer indentation level (filament_topology_diagnostics.py, line 50)
    ds_shot = xr.open_dataset(ds_shot_path)
    ds_eq_time = ds_shot.sel(idx=eq_time_idx)

    # Generate equilibrium field file
    eq_data = normalize_eq_field_dataset(ds_eq_time)
    geqdsk = build_geqdsk(eq_data)
    # cocos = detect_cocos(geqdsk) # Cocos check for diagnostic purposes
    geqdsk_converted = convert_cocos(geqdsk, cocos_target=1)
    eq_field = EquilibriumField(geqdsk_converted)

    # Output containers
    filament_lists = []
    current_lists = []
    direct_response_dss = []
    for mode in modes if np.size(modes[0]) > 1 else [modes]:
        # Generate equilibrium field geometetry
        tracer = EquilibriumFilamentTracer(
            mode[0],
            mode[1],
            eq_field=eq_field,
            base_num_points=config.direct_response_filament_points,
        )

        # Run Tracer
        # Output cartesian for ThinCurr/Biot-Savart mutual inductance calculation
        filament_list, current_list = tracer.get_filament_list(
            num_filaments=config.direct_response_filaments,
            coordinate_system="cartesian",
            trace_type=EquilibriumFilamentTracer.TraceType.FIELD
        )

        filament_lists.append(filament_list)
        current_lists.append(current_list)

        # Extrac sensor details
        sensor_details = build_sensor_details_compat(
            ds_eq_time, sensor_set_name="mirnov_custom"
        )

        # Run Biot-Savart to get direct response
        direct_response = direct_response_biot_savart(
            sensor_details, filament_list, current_list
        )

        # Create xarray Dataset for direct response
        direct_response_ds = xr.Dataset(
            data_vars={
                "direct_response_real": (["sensor_idx"], direct_response.real),
                "direct_response_imag": (["sensor_idx"], direct_response.imag),
            },
            coords={"sensor_idx": sensor_details["sensor_idx"].data},
            attrs={"sensor_names": sensor_details["sensor_name"].data},
        )
        direct_response_dss.append(direct_response_ds)

    plot_equilibrium_diagnostics(
        eq_field, geqdsk_converted, filament_lists, modes, doSave, ds_shot_path
    )
    print(direct_response_dss)


###############################################################33
def plot_equilibrium_diagnostics(
    eq_field, eqdsk, filament_lists, modes, doSave, ds_shot_path
):
    # Build diagnostic plots for traver/equilibrium

    fig, ax = plt.subplots(1, 3, layout="constrained", figsize=(9, 3))

    ax[0].plot(eq_field.psi_grid, eq_field.qpsi(eq_field.psi_grid))
    ax[0].grid()
    ax[0].set_xlabel(r"$\psi$")
    ax[0].set_ylabel(r"$q(\psi)$")
    ax[0].set_title("q profile")

    im = ax[1].contour(eqdsk.r_grid, eqdsk.z_grid, eqdsk.psirz, levels=20)
    fig.colorbar(im, ax=ax[1], label=r"$\psi$")
    ax[1].grid()

    for mode in modes if np.size(modes[0]) > 1 else [modes]:
        psi_ind = np.argmin(
            np.abs(eq_field.qpsi(eq_field.psi_grid) - mode[0] / mode[1])
        )
        ax[1].contour(
            eqdsk.r_grid,
            eqdsk.z_grid,
            eqdsk.psirz,
            levels=[eq_field.psi_grid[psi_ind]],
            colors="k",
            linewidths=2,
        )

    # Plot filament geometry on top of equilibrium
    # Filament list is in cartesian
    for ind, filament_list in enumerate(filament_lists):
        ax[1].plot(
            np.sqrt(filament_list[0][:, 0] ** 2 + filament_list[0][:, 1] ** 2),
            filament_list[0][:, 2],
            "*",
            label="q=" + str(modes[ind][0]) + "/" + str(modes[ind][1]),
        )
    ax[1].legend(fontsize=8)
    ax[1].set_xlabel("R [m]")
    ax[1].set_ylabel("Z [m]")
    ax[1].set_title("Filaments and internal equilibrium")

    q_grid = eq_field.qpsi(eqdsk.psirz)

    im = ax[2].contour(
        eqdsk.r_grid, eqdsk.z_grid, q_grid, levels=[0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    )
    fig.colorbar(im, ax=ax[2], label="q")
    ax[2].grid()
    ax[2].set_xlabel("R [m]")
    ax[2].set_ylabel("Z [m]")
    ax[2].set_title("q profile on gEQDSK")
    for mode in modes if np.size(modes[0]) > 1 else [modes]:
        ax[2].contour(
            eqdsk.r_grid,
            eqdsk.z_grid,
            q_grid,
            levels=[mode[0] / mode[1]],
            colors="k",
            linewidths=2,
        )

    if doSave:
        fig_path = rf"{doSave}Diagnostic_Equilibria_{ds_shot_path.split('/')[-1].split('.')[0]}.pdf"
        plt.savefig(fig_path, transparent=True, bbox_inches="tight")
        print(f"Saved equilibrium diagnostics plot to {fig_path}")
    plt.show()


####################################################
if __name__ == "__main__":
    # ds_shot_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/179118.nc"
    # ds_shot_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/1120906030.nc"
    # eq_time_idx = 1200#23000
    # ds_shot_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/175028.nc"
    ds_shot_path = "/fusion/projects/disruption_warning/data/tmdb/laszlo_compare/IDA_shots/174956.nc"
    eq_time_idx = 1500  # 23000
    modes = [(3, 2), (4, 1)]

    doSave = "../Synthetic_Mirnov/output_plots/"
    plot_filament_geometry(ds_shot_path, eq_time_idx, modes, doSave)

    print("finished")
