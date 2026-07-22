# theta/phi contour plotting for direct response mode surfaces
from diagnostic_header import (
    EquilibriumField,
    EquilibriumFilamentTracer,
    build_geqdsk,
    build_sensor_details_compat,
    config,
    convert_cocos,
    direct_response_biot_savart,
    mtri,
    normalize_eq_field_dataset,
    np,
    os,
    plt,
    xr,
)

RAD2DEG = 180 / np.pi


def build_equilibrium_field(ds_eq_time: xr.Dataset) -> EquilibriumField:
    eq_data = normalize_eq_field_dataset(ds_eq_time)
    geqdsk = build_geqdsk(eq_data)
    # cocos = detect_cocos(geqdsk) # Cocos check for diagnostic purposes
    geqdsk_converted = convert_cocos(geqdsk, cocos_target=1)
    return EquilibriumField(geqdsk_converted)


def compute_direct_response_for_mode(mode, eq_field, ds_eq_time):
    tracer = EquilibriumFilamentTracer(
        mode[0],
        mode[1],
        eq_field=eq_field,
        base_num_points=config.direct_response_filament_points,
    )

    filament_list, current_list = tracer.get_filament_list(
        num_filaments=config.direct_response_filaments,
        coordinate_system="cartesian",
    )

    sensor_details = build_sensor_details_compat(
        ds_eq_time, sensor_set_name="mirnov_custom"
    )
    direct_response = direct_response_biot_savart(
        sensor_details, filament_list, current_list
    )

    sensor_phi_deg = sensor_details["sensor_phi"].data
    sensor_phi = np.radians(sensor_phi_deg) % (2 * np.pi)

    sensor_theta = np.arctan2(
        sensor_details["sensor_Z"].data,
        sensor_details["sensor_R"].data - ds_eq_time.rmaxis.values,
    )
    sensor_names = (
        sensor_details["sensor_name"].data if "sensor_name" in sensor_details else None
    )

    return direct_response, sensor_phi, sensor_theta, sensor_names


def fill_plane_contour_values(sensor_phi, sensor_theta, values):
    phi_grid = np.linspace(0, 2 * np.pi, 200)
    theta_min = np.min(sensor_theta)
    theta_max = np.max(sensor_theta)
    theta_span = max(1e-3, theta_max - theta_min)
    theta_grid = np.linspace(
        theta_min - 0.25 * theta_span, theta_max + 0.25 * theta_span, 120
    )
    phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)

    triang = mtri.Triangulation(sensor_phi, sensor_theta)
    interp = mtri.LinearTriInterpolator(triang, values)
    grid_values = interp(phi_mesh, theta_mesh)

    if np.any(np.isnan(grid_values)):
        points = np.vstack((sensor_phi, sensor_theta)).T
        flat_points = np.vstack((phi_mesh.ravel(), theta_mesh.ravel())).T
        dist2 = np.sum((flat_points[:, None, :] - points[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(dist2, axis=1)
        flat_values = grid_values.ravel()
        nan_mask = np.isnan(flat_values)
        flat_values[nan_mask] = values[nearest[nan_mask]]
        grid_values = flat_values.reshape(grid_values.shape)

    return phi_mesh, theta_mesh, grid_values


def plot_response_contours_for_mode(
    mode, direct_response, sensor_phi, sensor_theta, sensor_names, save_path
):
    real_values = direct_response.real
    imag_values = direct_response.imag

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        f"Mode m={mode[0]}, n={mode[1]} : Direct Response in theta/phi", fontsize=14
    )

    for ax, values, label in zip(axes, [real_values, imag_values], ["Real", "Imag"]):
        phi_mesh, theta_mesh, grid_values = fill_plane_contour_values(
            sensor_phi, sensor_theta, values
        )
        contour = ax.contourf(
            phi_mesh * RAD2DEG,
            theta_mesh * RAD2DEG,
            grid_values,
            levels=20,
            cmap="RdBu_r",
            extend="both",
            zorder=-5,
        )
        ax.scatter(sensor_phi, sensor_theta, c="k", s=60, zorder=5)

        for ind, name in enumerate(sensor_names):
            ax.text(
                sensor_phi[ind] * RAD2DEG,
                sensor_theta[ind] * RAD2DEG,
                str(name),
                color="black",
                fontsize=8,
                ha="center",
                va="bottom",
            )

        cbar = fig.colorbar(contour, ax=ax, orientation="vertical", pad=0.03)
        cbar.set_label(f"Direct response {label} [T]")
        ax.set_xlabel(r"$\phi$ [deg]")
        ax.set_ylabel(r"$\theta = \arctan(Z/R)$ [deg]")
        ax.set_title(f"{label}({mode[0]},{mode[1]})")
        ax.grid(True)
        ax.set_xlim(0, 2 * np.pi * RAD2DEG)
        theta_max = np.max(sensor_theta) * RAD2DEG
        theta_min = np.min(sensor_theta) * RAD2DEG
        buffer = max(0.01, 0.1 * (theta_max - theta_min))
        ax.set_ylim(theta_min - buffer, theta_max + buffer)
        ax.set_rasterization_zorder(-1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, transparent=True, bbox_inches="tight")
    # Also save as png for easy viewing
    fig.savefig(
        save_path.replace(".pdf", ".png"), transparent=True, bbox_inches="tight"
    )
    plt.show()

    print(f"Saved mode contour figure to {save_path}")

    plt.close(fig)


def plot_phase_geometric_comparison(
    mode,
    signal_complex,
    sensor_phi,
    sensor_theta,
    sensor_names,
    save_path,
    timepoint_s,
    frequency_khz,
):
    m, n = mode
    signal_phase = np.angle(signal_complex)

    # Poloidal plot: Corrected phase = phase - n * phi vs theta
    # Subtract n * phi to look for the m*theta part
    pol_corrected_phase = signal_phase - n * sensor_phi
    # Wrap to [-pi, pi]
    pol_corrected_phase = (pol_corrected_phase + np.pi) % (2 * np.pi) - np.pi

    # Toroidal plot: Bands of theta
    # Group sensors by theta with 10 degree tolerance
    tol = 10 / RAD2DEG

    # Sort by theta to make grouping easier
    sorted_idx = np.argsort(sensor_theta)
    s_theta = sensor_theta[sorted_idx]
    s_phi = sensor_phi[sorted_idx]
    s_phase = signal_phase[sorted_idx]
    s_names = (
        sensor_names[sorted_idx] if sensor_names is not None else [None] * len(s_theta)
    )

    all_rel_phi = []
    all_rel_phase = []
    all_synth_rel_phase = []
    all_band_labels = []

    if len(s_theta) > 0:
        # Improved banding: group sensors that have VERY similar theta
        bands = []
        if len(s_theta) > 0:
            current_band = [0]
            for i in range(1, len(s_theta)):
                if s_theta[i] - s_theta[current_band[0]] < tol:
                    current_band.append(i)
                else:
                    bands.append(current_band)
                    current_band = [i]
            bands.append(current_band)

        print("\n--- Diagnostic Numeric Output ---")
        for band in bands:
            if len(band) < 2:
                continue  # Skip bands with single sensors as requested

            # Use the first sensor in the band as the reference for this plot
            # This ensures that for the reference sensor (Rel Phi = 0), the Rel Phase is also 0.
            ref_idx = band[0]
            ref_phi = s_phi[ref_idx]

            # Subtract the m*theta contribution from the whole band
            s_phase_toroidal_band = s_phase[band] - m * s_theta[band]

            # Use the average theta of the band for labeling/coloring
            avg_theta = np.mean(s_theta[band])
            avg_theta_deg = avg_theta * RAD2DEG

            # For diagnostic printing
            if abs(avg_theta_deg) < 15:
                print(
                    f"\nToroidal Midplane Band (avg_theta ~ {avg_theta_deg:.1f} deg):"
                )
                print(
                    f"{'Sensor':<15} | {'Rel Phi (deg)':>12} | {'Rel Phase (deg)':>15} | {'Expected (n=1)':>15} | {'Diff':>8}"
                )

            # Define synthetic phase for this band, zeroed at the same reference
            synthetic_phase_band = m * s_theta[band] + n * s_phi[band]
            synth_toroidal_band = synthetic_phase_band - m * s_theta[band]

            for i_in_band, idx in enumerate(band):
                # Geometric toroidal angle relative to band reference
                rel_p = (s_phi[idx] - ref_phi + np.pi) % (2 * np.pi) - np.pi

                # Signal phase (m*theta corrected) zeroed to the reference sensor
                # rel_sig = (Phase - m*theta)_idx - (Phase - m*theta)_ref
                rel_sig = (
                    s_phase_toroidal_band[i_in_band]
                    - (s_phase[ref_idx] - m * s_theta[ref_idx])
                    + np.pi
                ) % (2 * np.pi) - np.pi

                # Same for synthetic
                rel_sig_s = (
                    synth_toroidal_band[i_in_band] - (synth_toroidal_band[0]) + np.pi
                ) % (2 * np.pi) - np.pi

                expected = (n * rel_p + np.pi) % (2 * np.pi) - np.pi
                diff = (rel_sig - expected + np.pi) % (2 * np.pi) - np.pi

                if abs(avg_theta_deg) < 15:
                    print(
                        f"{str(s_names[idx]):<15} | {rel_p * RAD2DEG:>12.1f} | {rel_sig * RAD2DEG:>15.1f} | {expected * RAD2DEG:>15.1f} | {diff * RAD2DEG:>8.1f}"
                    )

                all_rel_phi.append(rel_p)
                all_rel_phase.append(rel_sig)
                all_synth_rel_phase.append(rel_sig_s)
                all_band_labels.append(avg_theta)

    # Now plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        f"Measured Phase Comparison (t={timepoint_s}s, f={frequency_khz}kHz) for Mode m={m}, n={n}",
        fontsize=14,
    )

    # Subplot 1: Poloidal
    # Automatically find the toroidal angle with the greatest number of sensors
    phi_bins = np.linspace(0, 360, 73)  # 5 degree bins
    hist, bin_edges = np.histogram(sensor_phi * RAD2DEG, bins=phi_bins)
    max_bin = np.argmax(hist)
    best_phi = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2

    print(
        f"\nDetected best poloidal array at phi ~ {best_phi:.1f} deg ({hist[max_bin]} sensors)"
    )

    pol_mask = np.abs(sensor_phi * RAD2DEG - best_phi) < 10

    axes[0].scatter(
        sensor_theta[pol_mask] * RAD2DEG,
        pol_corrected_phase[pol_mask] * RAD2DEG,
        c="blue",
        alpha=0.9,
        s=80,
        edgecolors="k",
        label=f"poloidal array (phi ~ {best_phi:.1f})",
    )

    # Synthetic Poloidal Points
    synth_pol_phase = m * sensor_theta[pol_mask]
    # Align synthetic phase to the mean of the measured phase to overlap
    alignment_offset = np.median(
        (pol_corrected_phase[pol_mask] - synth_pol_phase + np.pi) % (2 * np.pi) - np.pi
    )
    axes[0].scatter(
        sensor_theta[pol_mask] * RAD2DEG,
        ((synth_pol_phase + alignment_offset + np.pi) % (2 * np.pi) - np.pi) * RAD2DEG,
        marker="x",
        color="red",
        s=40,
        alpha=0.7,
        label="pure helical (m,n)",
    )

    # Poloidal Diagnostic: fixed phi ~ best_phi
    print(f"\nPoloidal Diagnostic (phi ~ {best_phi:.1f} deg, n={n}):")
    print(
        f"{'Sensor':<15} | {'Phi':>6} | {'Theta':>7} | {'Phase':>7} | {'S-n*phi':>9} | {'Local m':>7}"
    )
    pol_indices = [i for i in np.where(pol_mask)[0]]
    # Sort for local m calculation
    pol_indices = sorted(pol_indices, key=lambda idx: sensor_theta[idx])

    for idx, i in enumerate(pol_indices):
        loc_m = ""
        if idx > 0:
            prev_i = pol_indices[idx - 1]
            d_phi = (sensor_phi[i] - sensor_phi[prev_i] + np.pi) % (2 * np.pi) - np.pi
            d_theta = (sensor_theta[i] - sensor_theta[prev_i] + np.pi) % (
                2 * np.pi
            ) - np.pi
            # Use original phase for local m calculation
            d_phase = (
                np.angle(signal_complex[i]) - np.angle(signal_complex[prev_i]) + np.pi
            ) % (2 * np.pi) - np.pi
            if abs(d_theta) > 1e-3:
                # Local m = (d_phase - n*d_phi)/d_theta
                loc_m = f"{(d_phase - n * d_phi) / d_theta:.2f}"

        print(
            f"{str(sensor_names[i]):<15} | {sensor_phi[i] * RAD2DEG:>6.1f} | {sensor_theta[i] * RAD2DEG:>7.1f} | {np.angle(signal_complex[i]) * RAD2DEG:>7.1f} | {pol_corrected_phase[i] * RAD2DEG:>9.1f} | {loc_m:>7}"
        )

    # Plot guide lines for m
    if np.any(pol_mask):
        theta_range = np.linspace(
            np.min(sensor_theta[pol_mask]), np.max(sensor_theta[pol_mask]), 100
        )
        ref_theta = sensor_theta[pol_indices[len(pol_indices) // 2]]
        ref_phase = pol_corrected_phase[pol_indices[len(pol_indices) // 2]]
        axes[0].plot(
            theta_range * RAD2DEG,
            ((m * (theta_range - ref_theta) + ref_phase + np.pi) % (2 * np.pi) - np.pi)
            * RAD2DEG,
            "r--",
            alpha=0.8,
            label=f"m={m} slope",
        )

    axes[0].set_xlabel(r"$\theta = \arctan(Z/R-R_{axis})$ [deg]")
    axes[0].set_ylabel(r"Signal Phase - $n\phi$ [deg]")
    axes[0].set_title("Poloidal Phase Structure")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    # Subplot 2: Toroidal
    if len(all_rel_phi) > 0:
        rel_phi_deg = np.array(all_rel_phi) * RAD2DEG
        rel_phase_deg = np.array(all_rel_phase) * RAD2DEG
        band_thetas = np.array(all_band_labels) * RAD2DEG

        # Filter for midplane (avg_theta ~ 0)
        midplane_mask = np.abs(band_thetas) < 15

        if np.any(midplane_mask):
            axes[1].scatter(
                rel_phi_deg[midplane_mask],
                rel_phase_deg[midplane_mask],
                c="green",
                alpha=0.9,
                s=80,
                edgecolors="k",
                label="midplane array (theta~0)",
            )

        # Also plot all bands with low alpha to see the remaining scatter
        axes[1].scatter(
            rel_phi_deg[~midplane_mask],
            rel_phase_deg[~midplane_mask],
            c="gray",
            alpha=0.2,
            s=40,
            edgecolors="none",
            label="other bands",
        )

        # Add Synthetic Phase Overlay (Theoretical m, n)
        if len(all_synth_rel_phase) == len(all_rel_phi):
            axes[1].scatter(
                rel_phi_deg,
                np.array(all_synth_rel_phase) * RAD2DEG,
                marker="x",
                color="red",
                s=20,
                alpha=0.5,
                label="pure helical (m,n)",
            )

        # Plot guide lines for n
        phi_range_deg = np.linspace(-180, 180, 100)
        axes[1].plot(
            phi_range_deg,
            (n * phi_range_deg + 180) % 360 - 180,
            "r--",
            alpha=0.8,
            label=f"n={n} line",
        )

        axes[1].set_xlabel(r"Relative $\phi$ [deg]")
        axes[1].set_ylabel("Relative Phase [deg]")
        axes[1].set_title("Toroidal Phase Structure (Banded)")
        axes[1].set_xlim(-180, 180)
        axes[1].set_ylim(-180, 180)
        axes[1].legend(fontsize=8)
    axes[1].grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, transparent=True, bbox_inches="tight")
    # Also save as png for easy viewing
    fig.savefig(
        save_path.replace(".pdf", ".png"), transparent=True, bbox_inches="tight"
    )
    plt.show()
    print(f"Saved measured phase comparison figure to {save_path}")
    plt.close(fig)


def plot_modes_theta_phi(
    ds_shot_path, eq_time_s, timepoint_s, frequency_khz, modes, output_dir
):
    ds_shot = xr.open_dataset(ds_shot_path)
    eq_time_idx = np.argmin(np.abs(ds_shot["time"].data - eq_time_s))
    ds_eq_time = ds_shot.sel(idx=eq_time_idx)
    eq_field = build_equilibrium_field(ds_eq_time)

    # Extract real signal for comparison at the specified timepoint and frequency
    t_idx_fft = np.argmin(np.abs(ds_shot["time"].data - timepoint_s))
    f_idx_fft = np.argmin(np.abs(ds_shot["frequency"].data - frequency_khz * 1e3))

    # Use the same sensor subset logic as the tracer (mirnov_custom)
    sensor_details = build_sensor_details_compat(
        ds_eq_time, sensor_set_name="mirnov_custom"
    )
    real_signal = None
    if "sensor_name" in sensor_details:
        target_names = sensor_details.sensor_name.data
        ds_names = ds_shot.sensor_name.data
        # Match names to indices to account for possible filtering in sensor_details
        name_to_idx = {name: i for i, name in enumerate(ds_names)}
        indices = [name_to_idx[name] for name in target_names if name in name_to_idx]

        real_fft = ds_shot.mirnov_fft_real.isel(
            idx=t_idx_fft, frequency_idx=f_idx_fft, sensor_idx=indices
        ).values
        imag_fft = ds_shot.mirnov_fft_imag.isel(
            idx=t_idx_fft, frequency_idx=f_idx_fft, sensor_idx=indices
        ).values
        real_signal = real_fft + 1j * imag_fft

    for mode in modes if np.ndim(modes) == 2 else [modes]:
        direct_response, sensor_phi, sensor_theta, sensor_names = (
            compute_direct_response_for_mode(mode, eq_field, ds_eq_time)
        )
        save_path = os.path.join(
            output_dir,
            f"mode_m{mode[0]}_n{mode[1]}_theta_phi_contours_{ds_shot_path.split('/')[-1].split('.')[0]}.pdf",
        )
        plot_response_contours_for_mode(
            mode, direct_response, sensor_phi, sensor_theta, sensor_names, save_path
        )

        if real_signal is not None:
            phase_save_path = os.path.join(
                output_dir,
                f"mode_m{mode[0]}_n{mode[1]}_measured_phase_comparison_{ds_shot_path.split('/')[-1].split('.')[0]}.pdf",
            )
            plot_phase_geometric_comparison(
                mode,
                real_signal,
                sensor_phi,
                sensor_theta,
                sensor_names,
                phase_save_path,
                timepoint_s,
                frequency_khz,
            )


if __name__ == "__main__":
    # ds_shot_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/179118.nc"
    # eq_time_idx = 23000

    # ds_shot_path = "/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/1120906030.nc"
    # eq_time_idx = 1200#23000
    shot = 174956
    ds_shot_path = f"/mnt/home/rianc/Documents/TARS/tars/scratch/input_data/{shot}.nc"
    eq_time_s = 2.30  # 23000
    timepoint_s = 2.50
    frequency_khz = 10
    modes = [(2, 1), (3, 1)]
    output_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "output_plots", "theta_phi_contours"
        )
    )
    plot_modes_theta_phi(
        ds_shot_path, eq_time_s, timepoint_s, frequency_khz, modes, output_dir
    )
