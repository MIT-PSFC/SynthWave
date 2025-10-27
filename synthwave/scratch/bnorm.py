import os
import numpy as np
import matplotlib.pyplot as plt

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.meshing import (
    build_torus_bnorm_grid,
    ThinCurr_periodic_toroid,
)

from synthwave import PACKAGE_ROOT

EXAMPLE_DIR = os.path.join(PACKAGE_ROOT, "..", "thincurr_scratch", "bnorm")

MAJOR_RADIUS = 1
MINOR_RADIUS = 0.3


def create_circular_bnorm(filename, R0, Z0, a, n, m, npts=200):
    theta_vals = np.linspace(0.0, 2 * np.pi, npts, endpoint=False)
    with open(filename, "w+") as fid:
        fid.write("{0} {1}\n".format(npts, n))
        for theta in theta_vals:
            fid.write(
                "{0} {1} {2} {3}\n".format(
                    R0 + a * np.cos(theta),
                    Z0 + a * np.sin(theta),
                    np.cos(m * theta),
                    np.sin(m * theta),
                )
            )


if __name__ == "__main__":
    if not os.path.exists(EXAMPLE_DIR):
        os.makedirs(EXAMPLE_DIR)

    circular_bnorm_file = os.path.join(EXAMPLE_DIR, "circular_bnorm.dat")
    create_circular_bnorm(circular_bnorm_file, MAJOR_RADIUS, 0.0, MINOR_RADIUS, 1, 2)
    ntheta = 40
    nphi = 80
    r_grid, bnorm, nfp = build_torus_bnorm_grid(
        circular_bnorm_file, ntheta, nphi, resample_type="theta", use_spline=False
    )
    plasma_mode = ThinCurr_periodic_toroid(r_grid, nfp, ntheta, nphi)
    plasma_mode_file = os.path.join(EXAMPLE_DIR, "plasma_mode.h5")
    plasma_mode.write_to_file(plasma_mode_file)

    myOFT = OFT_env(nthreads=4)
    tw_mode = ThinCurr(myOFT)
    tw_mode.setup_model(mesh_file=plasma_mode_file)
    tw_mode.setup_io(basepath=EXAMPLE_DIR)

    # Compute self-inductance
    tw_mode.compute_Lmat()
    Lmat_new = plasma_mode.condense_matrix(tw_mode.Lmat)
    Linv = np.linalg.inv(Lmat_new)

    bnorm_flat = bnorm.reshape((2, bnorm.shape[1] * bnorm.shape[2]))
    # Get surface flux from normal field
    flux_flat = bnorm_flat.copy()

    flux_flat[0, plasma_mode.r_map] = tw_mode.scale_va(bnorm_flat[0, plasma_mode.r_map])
    flux_flat[1, plasma_mode.r_map] = tw_mode.scale_va(bnorm_flat[1, plasma_mode.r_map])
    tw_mode.save_scalar(bnorm_flat[0, plasma_mode.r_map], "Bn_c")
    tw_mode.save_scalar(bnorm_flat[1, plasma_mode.r_map], "Bn_s")
    output_full = np.zeros((2, tw_mode.nelems))
    output_unique = np.zeros((2, Linv.shape[0]))
    for j in range(2):
        output_unique[j, :] = np.dot(Linv, plasma_mode.nodes_to_unique(flux_flat[j, :]))
        output_full[j, :] = plasma_mode.expand_vector(output_unique[j, :])

    tw_mode.save_current(output_full[0, :], "Jc")
    tw_mode.save_current(output_full[1, :], "Js")
    tw_mode.build_XDMF()

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].contour(plasma_mode.unique_to_nodes_2D(output_unique[0, :]), 10)
    ax[0, 1].contour(plasma_mode.unique_to_nodes_2D(output_unique[1, :]), 10)
    ax[1, 0].contour(bnorm[0, :, :].transpose(), 10)
    ax[1, 1].contour(bnorm[1, :, :].transpose(), 10)

    fig.savefig(os.path.join(EXAMPLE_DIR, "circular_bnorm_solution.png"), dpi=300)
