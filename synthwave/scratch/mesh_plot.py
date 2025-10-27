import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from synthwave import PACKAGE_ROOT

from OpenFUSIONToolkit.ThinCurr.meshing import ThinCurr_periodic_toroid

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markeredgewidth"] = 2


EXAMPLE_DIR = os.path.join(PACKAGE_ROOT, "..", "thincurr_scratch", "mesh_plot")

MAJOR_RADIUS = 1
MINOR_RADIUS = 0.3


def create_torus_mesh(R0, a):
    # Create r_grid: [nphi, ntheta, 3] array defining the surface of one field period
    nfp = 1
    ntheta = 40
    nphi = 80

    # Create poloidal and toroidal angle grids
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)

    # Create meshgrid
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")

    # Calculate Cartesian coordinates for the torus surface
    # r_grid shape: [nphi, ntheta, 3]
    r_grid = np.zeros((nphi, ntheta, 3))
    r_grid[:, :, 0] = (R0 + a * np.cos(theta_grid)) * np.cos(phi_grid)  # x
    r_grid[:, :, 1] = (R0 + a * np.cos(theta_grid)) * np.sin(phi_grid)  # y
    r_grid[:, :, 2] = a * np.sin(theta_grid)  # z

    torus_mesh = ThinCurr_periodic_toroid(r_grid, nfp, ntheta, nphi)
    return torus_mesh


def plot_original_mesh(mesh, fig_file: str):
    fig = plt.figure(figsize=(8, 8))
    mesh.plot_mesh(fig)
    fig.savefig(fig_file, dpi=300)
    plt.close(fig)


def plot_model_mesh(mesh_file, fig_file: str):
    fig = plt.figure(figsize=(8, 8))

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
    fig.savefig(fig_file, dpi=300)
    plt.close(fig)


def example_plotting_meshes():
    # Making your own torus mesh
    torus_mesh = create_torus_mesh(MAJOR_RADIUS, MINOR_RADIUS)
    torus_mesh_file = os.path.join(EXAMPLE_DIR, "thincurr_torus_mesh.h5")
    torus_mesh.write_to_file(torus_mesh_file)
    plot_model_mesh(
        torus_mesh_file, fig_file=os.path.join(EXAMPLE_DIR, "thincurr_torus_mesh.png")
    )

    # Loading and plotting an existing mesh model
    cmod_mesh_file = os.path.join(
        PACKAGE_ROOT, "input_data", "cmod", "C_Mod_ThinCurr_VV-homology.h5"
    )
    plot_model_mesh(
        cmod_mesh_file, fig_file=os.path.join(EXAMPLE_DIR, "thincurr_cmod_mesh.png")
    )


if __name__ == "__main__":
    if not os.path.exists(EXAMPLE_DIR):
        os.makedirs(EXAMPLE_DIR)
    example_plotting_meshes()
