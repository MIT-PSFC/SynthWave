#!/bin/bash
#
# SynthWave Installation Script
# ========================
# Installs SynthWave and Python dependencies. No sudo required.
#
# Usage:
#   ./install.sh                    # Install dependencies
#   ./install.sh --clean            # Clean previous installation first
#   ./install.sh --help             # Show help
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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
        --clean) CLEAN=true; shift ;;
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

cd "$SCRIPT_DIR"

# Clean if requested
if $CLEAN; then
    header "Cleaning previous installation"
    rm -rf .venv .env setup_env.sh submodules/OpenFUSIONToolkit 2>/dev/null || true
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

# Create setup_env.sh
header "Creating environment files"
cat > "${SCRIPT_DIR}/setup_env.sh" << 'EOF'
#!/bin/bash
# Source this for manual Python usage: source setup_env.sh
SYNTHWAVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SYNTHWAVE_ROOT}/.venv/bin/activate" ]; then
    source "${SYNTHWAVE_ROOT}/.venv/bin/activate"
else
    echo "SynthWave virtual environment not found. Run ./install.sh or uv sync --dev first." >&2
fi
EOF
chmod +x "${SCRIPT_DIR}/setup_env.sh"
success "Created setup_env.sh"

# Install Python dependencies
header "Installing Python dependencies"
uv sync --dev
success "Python packages installed"
uv run pre-commit install

# Verify
header "Verifying installation"
uv run python -c "import synthwave" && success "synthwave" || warn "synthwave failed"
uv run python -c "from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov" 2>/dev/null \
    && success "OpenFUSIONToolkit" || warn "OpenFUSIONToolkit import failed"

# Done
header "Installation complete!"
