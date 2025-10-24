import numpy as np
import matplotlib.pyplot as plt

from synthwave.magnetic_geometry.filaments import FilamentTracer, ToroidalFilamentTracer


def plot_filaments_3d(
    tracer: FilamentTracer, fig_file: str, title: str = "Filament Traces in 3D"
):
    """
    Plot the filaments traced by the FilamentTracer in 3D with 4 different viewing angles.

    Args:
        tracer (FilamentTracer): The filament tracer object.
        fig_file (str): Output file path for the figure.
        title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(16, 12))

    # Create 4 subplots with different viewing angles
    views = [
        (221, "Default View", -60, 30),  # Default 3D view
        (222, "View along X axis", 0, 0),  # Looking from X
        (223, "View along Y axis", 90, 0),  # Looking from Y
        (224, "View along Z axis", 0, 90),  # Looking down from Z
    ]

    filament_ds = tracer.make_points_and_currents(
        num_filaments=3, coordinate_system="cartesian"
    )

    # Make a color scale where red is positive current and blue is negative current
    # and the colormap scale is between -1 and 1
    color_scale = plt.cm.bwr
    norm = plt.Normalize(
        vmin=-np.max(np.abs(filament_ds.current.values)),
        vmax=np.max(np.abs(filament_ds.current.values)),
    )
    color_scale = plt.cm.ScalarMappable(norm=norm, cmap=color_scale)

    # Calculate common axis limits
    max_range = (
        np.array(
            [
                filament_ds.x.max() - filament_ds.x.min(),
                filament_ds.y.max() - filament_ds.y.min(),
                filament_ds.z.max() - filament_ds.z.min(),
            ]
        ).max()
        / 2.0
    )
    xlim = (filament_ds.x.mean() - max_range, filament_ds.x.mean() + max_range)
    ylim = (filament_ds.y.mean() - max_range, filament_ds.y.mean() + max_range)
    zlim = (filament_ds.z.mean() - max_range, filament_ds.z.mean() + max_range)

    for subplot_idx, view_title, azim, elev in views:
        ax = fig.add_subplot(subplot_idx, projection="3d")

        for filament in filament_ds.filament.values:
            pts = np.array(
                [
                    filament_ds.x.sel(filament=filament).values,
                    filament_ds.y.sel(filament=filament).values,
                    filament_ds.z.sel(filament=filament).values,
                ]
            )
            current = filament_ds.current.sel(filament=filament).item()
            ax.plot(
                pts[0, :],
                pts[1, :],
                pts[2, :],
                color=color_scale.to_rgba(current),
                label=f"Filament {filament}, I={current:.2f} A",
            )

        ax.set_title(f"{title}\n{view_title}", fontsize=11)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        # Set the same limits for all subplots
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # Set the viewing angle
        ax.view_init(elev=elev, azim=azim)

    # Add a color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(color_scale, cax=cbar_ax)
    cbar.set_label("Filament Current [A]")

    fig.tight_layout()
    fig.savefig(fig_file)
    plt.close(fig)


if __name__ == "__main__":
    tracer = ToroidalFilamentTracer(3, 2, 1, 0, 0.5, 100)
    plot_filaments_3d(tracer, "filament_traces_3d.png", "Toroidal Filament Traces")
