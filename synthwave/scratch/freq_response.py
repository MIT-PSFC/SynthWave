import os


from synthwave import PACKAGE_ROOT
from synthwave.scratch.mesh_plot import create_torus_mesh
from synthwave.magnetic_geometry.filaments import ToroidalFilament
from synthwave.magnetic_geometry.utils import (
    cylindrical_to_cartesian,
)

EXAMPLE_DIR = os.path.join(PACKAGE_ROOT, "..", "thincurr_scratch", "freq_response")

MAJOR_RADIUS = 1
MINOR_RADIUS = 0.3


if __name__ == "__main__":
    # Goal: Plot response of icoils on a toroidal mesh
    if not os.path.exists(EXAMPLE_DIR):
        os.makedirs(EXAMPLE_DIR)

    # Create torus mesh if it does not exist
    torus_mesh_file = os.path.join(EXAMPLE_DIR, "thincurr_torus_mesh.h5")
    if not os.path.exists(torus_mesh_file):
        torus_mesh = create_torus_mesh(MAJOR_RADIUS, MINOR_RADIUS)
        torus_mesh.write_to_file(torus_mesh_file)

    # Create 'oft_in.xml' file with icoil definitions
    filament = ToroidalFilament(2, 1, MAJOR_RADIUS, 0, MINOR_RADIUS)
    filament_points_cylindrical, filament_etas = filament.trace(50)
    filament_points_cartesian = cylindrical_to_cartesian(filament_points_cylindrical)
