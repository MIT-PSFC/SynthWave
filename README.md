# SynthWave
Generating synthetic magnetic measurements of MHD modes with ThinCurr

Originally developed by Rian Chandra [here](https://github.com/chandrarn/Synthetic_Mirnov/tree/main/synthetic_mirnov_generation_tool_release)

## Installation Instructions

SynthWave is automatically installed as part of the TARS installation process. The TARS `install.sh` script will:

1. Download and install [OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) to `../OpenFUSIONToolkit`
2. Configure the Python bindings automatically
3. Install SynthWave and its dependencies via `uv sync`

If you want to use SynthWave standalone (outside of TARS), you need to:

1. Download and install [OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) v1.0.0-beta6
2. Extract it to `../OpenFUSIONToolkit` (relative to this directory)
3. Create a `pyproject.toml` in the OpenFUSIONToolkit Python directory:
```shell
cat > ../OpenFUSIONToolkit/python/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "openfusiontoolkit"
version = "1.0.0b6"
description = "OpenFUSIONToolkit Python bindings"
requires-python = ">=3.9"

[tool.setuptools]
packages = ["OpenFUSIONToolkit"]
EOF
```
4. Run `uv sync` from this directory

### Optional: VSCode Integration

For proper code highlighting in VSCode, add this to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/submodules/OpenFUSIONToolkit/python"
    ],
}
```