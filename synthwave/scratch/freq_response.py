import os
from synthwave import PACKAGE_ROOT
from synthwave.scratch.mesh_plot import create_torus_mesh
from synthwave.magnetic_geometry.filaments import ToroidalFilamentTracer
from synthwave.mirnov.prep_thincurr_input import gen_OFT_filament_and_eta_file

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
    toroidal_tracer = ToroidalFilamentTracer(2, 1, MAJOR_RADIUS, 0, MINOR_RADIUS)
    filament_list = toroidal_tracer.get_filament_list(num_filaments=10)

    oft_filament_file = os.path.join(EXAMPLE_DIR, "oft_in.xml")
    gen_OFT_filament_and_eta_file(
        working_directory=EXAMPLE_DIR,
        filament_list=filament_list,
        resistivity_list=[1e-6],
    )
