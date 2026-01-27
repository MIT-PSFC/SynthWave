#!/bin/bash
#
# SynthWave Installation Script
# ========================
# Installs SynthWave and dependencies. No sudo required.
#
# Usage:
#   ./install.sh                    # Interactive install
#   ./install.sh --clean            # Clean previous installation
#   ./install.sh --help             # Show help
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OFT_DIR="${SCRIPT_DIR}/submodules/OpenFUSIONToolkit"
OFT_VERSION="v1.0.0-beta6"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

info()    { echo -e "${BLUE}→ $1${NC}"; }
success() { echo -e "${GREEN}✓ $1${NC}"; }
warn()    { echo -e "${YELLOW}⚠ $1${NC}"; }
error()   { echo -e "${RED}✗ $1${NC}"; exit 1; }
header()  { echo -e "\n${BLUE}━━━ $1 ━━━${NC}\n"; }

# Parse arguments
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)         CLEAN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean           Remove previous installation first"
            echo "  --help            Show this help"
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# Clean if requested
if $CLEAN; then
    header "Cleaning previous installation"
    rm -rf "$OFT_DIR" .venv .env 2>/dev/null || true
    # Force remove if still exists
    [ -d "$OFT_DIR" ] && chmod -R u+w "$OFT_DIR" 2>/dev/null && rm -rf "$OFT_DIR"
    success "Clean complete"
fi

# Install uv if needed
header "Checking uv package manager"
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
success "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# Initialize submodules
header "Initializing submodules"
cd "$SCRIPT_DIR"
git submodule update --init --recursive
success "Submodules ready"

# Install OpenFUSIONToolkit
header "Installing OpenFUSIONToolkit"
if [ -d "$OFT_DIR/bin" ] && ls "$OFT_DIR/bin"/*.so &>/dev/null; then
    success "Already installed at $OFT_DIR"
else
    # Detect platform
    PLATFORM="Ubuntu_22_04-GNU-x86_64"
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        [[ "$ID" =~ ^(centos|rhel|rocky)$ ]] && PLATFORM="Centos_7-GNU-x86_64"
    fi
    
    info "Downloading OFT $OFT_VERSION ($PLATFORM)..."
    TARBALL="OpenFUSIONToolkit_${OFT_VERSION}-${PLATFORM}.tar.gz"
    URL="https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit/releases/download/${OFT_VERSION}/${TARBALL}"
    
    # Expected SHA256 checksums for v1.0.0-beta6
    # These should be verified against official release checksums
    declare -A CHECKSUMS=(
        ["OpenFUSIONToolkit_v1.0.0-beta6-Ubuntu_22_04-GNU-x86_64.tar.gz"]="VERIFY_CHECKSUM_BEFORE_USE"
        ["OpenFUSIONToolkit_v1.0.0-beta6-Centos_7-GNU-x86_64.tar.gz"]="VERIFY_CHECKSUM_BEFORE_USE"
    )
    
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    curl -L -o "$TEMP_DIR/$TARBALL" "$URL" || wget -O "$TEMP_DIR/$TARBALL" "$URL"
    
    # Verify checksum if available
    if [ "${CHECKSUMS[$TARBALL]}" != "VERIFY_CHECKSUM_BEFORE_USE" ] && [ -n "${CHECKSUMS[$TARBALL]}" ]; then
        info "Verifying checksum..."
        if command -v sha256sum &>/dev/null; then
            echo "${CHECKSUMS[$TARBALL]}  $TEMP_DIR/$TARBALL" | sha256sum -c - || error "Checksum verification failed!"
            success "Checksum verified"
        else
            warn "sha256sum not available, skipping checksum verification"
        fi
    else
        warn "No checksum available for $TARBALL - proceeding without verification"
        warn "Consider verifying the download manually or updating the CHECKSUMS array"
    fi
    
    info "Extracting..."
    tar -xzf "$TEMP_DIR/$TARBALL" -C "$TEMP_DIR"
    
    # Find the extracted directory
    EXTRACTED=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "OpenFUSIONToolkit_*" | head -1)
    if [ -z "$EXTRACTED" ]; then
        error "Failed to find extracted OpenFUSIONToolkit directory"
    fi
    
    mkdir -p "${SCRIPT_DIR}/submodules"
    rm -rf "$OFT_DIR"
    mv "$EXTRACTED" "$OFT_DIR"
    
    # Ensure pyproject.toml exists for Python bindings
    [ ! -f "$OFT_DIR/python/pyproject.toml" ] && cat > "$OFT_DIR/python/pyproject.toml" << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "openfusiontoolkit"
version = "1.0.0b6"
requires-python = ">=3.9"
dependencies = []

[tool.setuptools]
packages = ["OpenFUSIONToolkit"]
EOF
    
    trap - EXIT
    rm -rf "$TEMP_DIR"
    success "Installed to $OFT_DIR"
fi

# Create .env file for uv
header "Creating environment files"

# Detect HDF5 library path (needed for OpenFUSIONToolkit)
HDF5_LIB_PATH=""
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    # Check if HDF5 is already in LD_LIBRARY_PATH (from module)
    for path in $(echo "$LD_LIBRARY_PATH" | tr ':' '\n'); do
        if [ -f "$path/libhdf5.so.310" ] || [ -f "$path/libhdf5.so" ]; then
            HDF5_LIB_PATH="$path"
            break
        fi
    done
fi
# Also check common HPC locations
if [ -z "$HDF5_LIB_PATH" ]; then
    for path in /orcd/software/community/001/spack/pkg/hdf5/*/lib /usr/lib64 /usr/lib; do
        if [ -f "$path/libhdf5.so.310" ]; then
            HDF5_LIB_PATH="$path"
            break
        fi
    done 2>/dev/null
fi

if [ -n "$HDF5_LIB_PATH" ]; then
    info "Found HDF5 at: $HDF5_LIB_PATH"
fi

# Build LD_LIBRARY_PATH with current system path
DOTENV_LD_PATH="${OFT_DIR}/bin"
[ -n "${LD_LIBRARY_PATH:-}" ] && DOTENV_LD_PATH="${DOTENV_LD_PATH}:${LD_LIBRARY_PATH}"

cat > "${SCRIPT_DIR}/.env" << EOF
# SynthWave environment (loaded automatically by 'uv run')
# OpenFUSIONToolkit library path
LD_LIBRARY_PATH=${DOTENV_LD_PATH}
PATH=${OFT_DIR}/bin:/usr/local/bin:/usr/bin:/bin
EOF

# Add HDF5 to LD_LIBRARY_PATH if found
if [ -n "$HDF5_LIB_PATH" ]; then
    # Prepend HDF5 to the LD_LIBRARY_PATH line
    sed -i "s|^LD_LIBRARY_PATH=|LD_LIBRARY_PATH=${HDF5_LIB_PATH}:|" "${SCRIPT_DIR}/.env"
fi

# Create setup_env.sh
cat > "${SCRIPT_DIR}/setup_env.sh" << EOF
#!/bin/bash
# Source this for manual Python usage: source setup_env.sh
SYNTHWAVE_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export PATH="\${SYNTHWAVE_ROOT}/submodules/OpenFUSIONToolkit/bin:\${PATH}"
export LD_LIBRARY_PATH="\${SYNTHWAVE_ROOT}/submodules/OpenFUSIONToolkit/bin:\${LD_LIBRARY_PATH:-}"
EOF

# Add HDF5 path to setup_env.sh if found
if [ -n "$HDF5_LIB_PATH" ]; then
    cat >> "${SCRIPT_DIR}/setup_env.sh" << EOF
# HDF5 library path (detected during installation)
export LD_LIBRARY_PATH="${HDF5_LIB_PATH}:\${LD_LIBRARY_PATH}"
EOF
fi

# Install Python dependencies
header "Installing Python dependencies"
cd "$SCRIPT_DIR"

SYNC_ARGS=()

# Set up environment for uv run
export LD_LIBRARY_PATH="${OFT_DIR}/bin:${LD_LIBRARY_PATH:-}"

uv sync "${SYNC_ARGS[@]}" || uv sync
success "Python packages installed"
uv run pre-commit install

# Verify
header "Verifying installation"
uv run python -c "import synthwave" && success "synthwave" || warn "synthwave failed"
uv run python -c "from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov" 2>/dev/null \
    && success "OpenFUSIONToolkit" || warn "OpenFUSIONToolkit (may need LD_LIBRARY_PATH)"

# Done
header "Installation complete!"
