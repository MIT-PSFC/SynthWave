# SynthWave
Generating synthetic magnetic measurements of MHD modes with ThinCurr

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