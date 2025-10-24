import numpy as np
import matplotlib.pyplot as plt

from synthwave.magnetic_geometry.filaments import FilamentTracer, ToroidalFilamentTracer


def plot_filaments_3d(
    tracer: FilamentTracer, fig_file: str, title: str = "Filament Traces in 3D"
):
    """
    Plot the filaments traced by the FilamentTracer in 3D.

    Args:
        tracer (FilamentTracer): The filament tracer object.
        title (str): Title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    filament_ds = tracer.make_points_and_currents(
        num_filaments=1, coordinate_system="cartesian"
    )

    # Make a color scale where red is positive current and blue is negative current
    # and the colormap scale is between -1 and 1
    color_scale = plt.cm.bwr
    norm = plt.Normalize(
        vmin=-np.max(np.abs(filament_ds.current.values)),
        vmax=np.max(np.abs(filament_ds.current.values)),
    )
    color_scale = plt.cm.ScalarMappable(norm=norm, cmap=color_scale)

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

    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # Make the axes all have the same limits, based on the one which needs to be largest
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
    ax.set_xlim((filament_ds.x.mean() - max_range, filament_ds.x.mean() + max_range))
    ax.set_ylim((filament_ds.y.mean() - max_range, filament_ds.y.mean() + max_range))
    ax.set_zlim((filament_ds.z.mean() - max_range, filament_ds.z.mean() + max_range))

    fig.savefig(fig_file)
    plt.close(fig)


if __name__ == "__main__":
    tracer = ToroidalFilamentTracer(3, 2, 1, 0, 0.3, 100)
    plot_filaments_3d(tracer, "filament_traces_3d.png", "Toroidal Filament Traces")
