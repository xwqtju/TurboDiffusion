#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/b200/build_offline_bundle.sh OUTPUT_DIR [options]

Build a Linux x86_64 / Python 3.11 / CUDA 12.8 wheelhouse for transfer to an
offline B200 machine. Run this on a networked Linux x86_64 CUDA 12.8 build host.

Options:
  --python COMMAND     Python 3.11 command (default: python3.11)
  --profile PROFILE   runtime or training (default: runtime)
  --with-flash-attn   Build the optional flash-attn wheel for SM100
  -h, --help          Show this message
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

OUTPUT_DIR="$1"
shift
PYTHON_BIN="python3.11"
PROFILE="runtime"
WITH_FLASH_ATTN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --with-flash-attn)
            WITH_FLASH_ATTN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "Bundle building requires a Linux x86_64 host." >&2
    exit 1
fi
if [[ "$PROFILE" != "runtime" && "$PROFILE" != "training" ]]; then
    echo "--profile must be runtime or training." >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Cannot find $PYTHON_BIN." >&2
    exit 1
fi
PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "The offline bundle requires Python 3.11; found $PYTHON_VERSION." >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA 12.8 nvcc is required to build the project wheel." >&2
    exit 1
fi
NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
if [[ "$NVCC_RELEASE" != "12.8" ]]; then
    echo "Expected CUDA 12.8; nvcc reports ${NVCC_RELEASE:-unknown}." >&2
    exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/turbodiffusion/ops/cutlass/include/cutlass/cutlass.h" ]]; then
    echo "CUTLASS submodule is missing. Initialize submodules before building." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
WHEELHOUSE="${OUTPUT_DIR}/wheelhouse"
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/td-b200-build.XXXXXX")"
BUILD_VENV="${BUILD_ROOT}/venv"
cleanup() {
    rm -rf "$BUILD_ROOT"
}
trap cleanup EXIT

if [[ -e "$WHEELHOUSE" ]]; then
    echo "Refusing to overwrite an existing wheelhouse in $OUTPUT_DIR." >&2
    exit 1
fi

"$PYTHON_BIN" -m venv "$BUILD_VENV"
# shellcheck disable=SC1091
source "${BUILD_VENV}/bin/activate"
python -m pip install --upgrade pip
mkdir -p "$WHEELHOUSE"

python -m pip download --dest "$WHEELHOUSE" \
    torch==2.8.0 torchvision==0.23.0 triton==3.4.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install the build stack inside the isolated build venv, then compile native
# wheels for the exact Python/CUDA/architecture combination being transferred.
python -m pip install --no-index --find-links "$WHEELHOUSE" \
    torch==2.8.0 torchvision==0.23.0 triton==3.4.0
python -m pip install setuptools==75.8.0 wheel==0.45.1 packaging==24.2 ninja==1.11.1.3
python -m pip wheel --wheel-dir "$WHEELHOUSE" \
    -r "${PROJECT_ROOT}/requirements/b200/${PROFILE}.txt"

if [[ "$WITH_FLASH_ATTN" -eq 1 ]]; then
    MAX_JOBS="${MAX_JOBS:-8}" TORCH_CUDA_ARCH_LIST="10.0" \
        python -m pip wheel --wheel-dir "$WHEELHOUSE" --no-build-isolation \
        -r "${PROJECT_ROOT}/requirements/b200/flash-attn.txt"
fi

TURBODIFFUSION_CUDA_ARCHS=100 MAX_JOBS="${MAX_JOBS:-8}" \
    python -m pip wheel --wheel-dir "$WHEELHOUSE" --no-build-isolation --no-deps "$PROJECT_ROOT"

python -m pip freeze > "${OUTPUT_DIR}/build-environment.txt"
(
    cd "$WHEELHOUSE"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

cat > "${OUTPUT_DIR}/BUNDLE_INFO.txt" <<EOF
TurboDiffusion B200 offline wheelhouse
Created: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Profile: ${PROFILE}
Python: $(python --version 2>&1)
NVCC: ${NVCC_RELEASE}
FlashAttention included: ${WITH_FLASH_ATTN}
Project commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD)
Project dirty: $(if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then echo yes; else echo no; fi)
EOF

echo "Offline wheelhouse created at: $WHEELHOUSE"
echo "Transfer the repository and $OUTPUT_DIR to the B200 host, verify SHA256SUMS, then run:"
echo "  scripts/b200/install.sh --offline $WHEELHOUSE --project wheel --profile $PROFILE"
