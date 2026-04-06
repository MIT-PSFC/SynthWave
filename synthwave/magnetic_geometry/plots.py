import h5py
import numpy as np
import matplotlib.pyplot as plt

from synthwave.magnetic_geometry.filaments import FilamentTracer


def _clean_fig(fig):
    # fig shows in Jupyter notebooks and is cleaned up in scripts
    try:
        from IPython.display import display as ipy_display

        ipy_display(fig)
    except ImportError:
        plt.show()
    plt.close(fig)


def plot_filaments_3d(
    tracer: FilamentTracer,
    title: str | None = "Filament Traces in 3D",
    fig_file: str | None = None,
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
    if fig_file is not None:
        fig.savefig(fig_file, dpi=300)
    # fig shows in Jupyter notebooks and is cleaned up in scripts
    _clean_fig(fig)


def plot_model_mesh(mesh_file, fig_file=None):
    # Read mesh data from HDF5 file for plotting
    with h5py.File(mesh_file, "r") as f:
        r = f["mesh/R"][:]  # Vertices
        lc = f["mesh/LC"][:]  # Triangles (1-indexed)
        reg = f["mesh/REG"][:]  # noqa: F841 Regions

    # Find maximum extent in each direction for setting equal aspect ratio
    max_range = (
        np.array(
            [
                r[:, 0].max() - r[:, 0].min(),
                r[:, 1].max() - r[:, 1].min(),
                r[:, 2].max() - r[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    # Plot using matplotlib
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(r[:, 0], r[:, 1], r[:, 2], triangles=lc - 1, alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(r[:, 0].mean() - max_range, r[:, 0].mean() + max_range)
    ax.set_ylim(r[:, 1].mean() - max_range, r[:, 1].mean() + max_range)
    ax.set_zlim(r[:, 2].mean() - max_range, r[:, 2].mean() + max_range)
    if fig_file is not None:
        fig.savefig(fig_file, dpi=300)
    # fig shows in Jupyter notebooks and is cleaned up in scripts
    _clean_fig(fig)
