# SynthWave
Generating synthetic magnetic measurements of arbitrary _m/n#_ MHD modes at fixed frequency and magnetic equilibrium, within an arbitrary conducting 
finite element mesh and arbitrary magnetic sensors, with ThinCurr (part of the [OpenFUSION toolkit](https://github.com/OpenFUSIONToolkit)).

Originally developed by Rian Chandra [here](https://github.com/chandrarn/Synthetic_Mirnov/tree/main/synthetic_mirnov_generation_tool_release)

## Installation Instructions

1. Download and install [OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit), and make note of the folder where you put it
2. Run the following commands to make a symlink and a pyproject.toml
```shell
ln -s /path/to/OpenFUSIONToolkit ./submodules/OpenFUSIONToolkit
cat > ./submodules/OpenFUSIONToolkit/python << 'EOF'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "openfusiontoolkit"
version = "1.0.0b6"
description = "Local OpenFUSIONToolkit vendorized package"
requires-python = ">=3.9"
readme = "README.md"

[tool.setuptools]
packages = ["OpenFUSIONToolkit"]
EOF
```
3. Run `uv sync`, and everything should install
4. (Optional) if you want to get proper highlighting in VSCode, add this to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/submodules/OpenFUSIONToolkit/python"
    ],
}
```

5. Additional python packages to be installed (through pip)
   
## Operation Instructions

The example autorun script in ```python generate_synthetic_mirnov_signals.py``` will launch a simulation run  using the example finite element mesh,
magnetic equilibrium, and sensor information packaged in the `input_data/` subfolder (`C_Mod_ThinCurr_VV-homology.h5`, `g1051202011.1000` and
`sensor_details_C_MOD_ALL.nc` respectively).

These sensors represent the high-frequency Mirnovs and vacuum vessel conducting surface for the Alcator C-Mod tokamak. Additional and more detailed
conducting meshes for the C-Mod, HBT-EP, DIII-D, and SPARC tokamaks can be provided to users with the appropriate permissions, on request. 

The simulation will use a fillamentary representation of the plasma on the appropriate rational surface, tracing out the helicity of the field lines,
oscillating at a fixed frequency (10kHz, in the example). The measured signal across the Mirnov probes will include the direct inductive coupling
to the ``plasma'' filaments, as well as coupling to the induced eddy current response in the conducting mesh.

Given the provided input files, the magnitude of the output signal across the sensor set should look like the below:

