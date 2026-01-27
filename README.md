# SynthWave
Generating synthetic magnetic measurements of arbitrary _m/n#_ MHD modes at fixed frequency and magnetic equilibrium, within an arbitrary conducting 
finite element mesh and arbitrary magnetic sensors, with ThinCurr (part of the [OpenFUSION toolkit](https://github.com/OpenFUSIONToolkit)).

Originally developed by Rian Chandra [here](https://github.com/chandrarn/Synthetic_Mirnov/tree/main/synthetic_mirnov_generation_tool_release)

## Installation Instructions

1. Run `./install.sh` and everything should install

This will create a local version of OpenFUSIONToolkit and make a uv-managed virtual environment.

2. (Optional) if you want to get proper highlighting in VSCode, add this to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/submodules/OpenFUSIONToolkit/python"
    ],
}
```
   
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


![Basic sensor signal response output](https://github.com/MIT-PSFC/SynthWave/blob/main/synthwave/mirnov/input_data/Sensor_Signals_m04_n02_f1.0e%2B01kHz_CMod.svg)

