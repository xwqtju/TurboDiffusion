#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/b200/install.sh [options]

Options:
  --venv PATH          Virtual environment path (default: .venv-b200)
  --python COMMAND     Python used to create the venv (default: python3.11)
  --profile PROFILE    runtime or training (default: runtime)
  --offline DIR        Install only from an offline wheelhouse
  --project MODE       editable or wheel (default: editable)
  --with-flash-attn    Install optional flash-attn 2.8.3
  --skip-verify        Do not run the post-install environment check
  -h, --help           Show this message

Examples:
  scripts/b200/install.sh --profile runtime
  scripts/b200/install.sh --profile training --venv /opt/venvs/turbodiffusion
  scripts/b200/install.sh --offline /mnt/wheelhouse --project wheel
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv-b200"
PYTHON_BIN="python3.11"
PROFILE="runtime"
OFFLINE_DIR=""
PROJECT_MODE="editable"
WITH_FLASH_ATTN=0
RUN_VERIFY=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --offline)
            OFFLINE_DIR="$2"
            shift 2
            ;;
        --project)
            PROJECT_MODE="$2"
            shift 2
            ;;
        --with-flash-attn)
            WITH_FLASH_ATTN=1
            shift
            ;;
        --skip-verify)
            RUN_VERIFY=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This installer must run on the Linux x86_64 B200 host." >&2
    exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "Expected x86_64, found $(uname -m)." >&2
    exit 1
fi
if [[ "$PROFILE" != "runtime" && "$PROFILE" != "training" ]]; then
    echo "--profile must be runtime or training." >&2
    exit 2
fi
if [[ "$PROJECT_MODE" != "editable" && "$PROJECT_MODE" != "wheel" ]]; then
    echo "--project must be editable or wheel." >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Cannot find $PYTHON_BIN. Install Python 3.11 or pass --python." >&2
    exit 1
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "The reproducible B200 profile requires Python 3.11; found $PYTHON_VERSION." >&2
    exit 1
fi

if [[ -n "$OFFLINE_DIR" ]]; then
    OFFLINE_DIR="$(cd "$OFFLINE_DIR" && pwd)"
    if [[ ! -d "$OFFLINE_DIR" ]]; then
        echo "Offline wheelhouse does not exist: $OFFLINE_DIR" >&2
        exit 1
    fi
fi

if [[ -e "$VENV_PATH" ]]; then
    echo "Refusing to reuse an existing environment: $VENV_PATH" >&2
    echo "Choose a new --venv path or archive and remove the old environment explicitly." >&2
    exit 1
fi

if [[ "$PROJECT_MODE" == "wheel" && -z "$OFFLINE_DIR" ]]; then
    echo "--project wheel requires --offline DIR containing a TurboDiffusion wheel." >&2
    exit 2
fi

if [[ "$PROJECT_MODE" == "editable" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "Editable install compiles CUDA code and requires CUDA 12.8 nvcc." >&2
        echo "Use --project wheel with a wheel built by build_offline_bundle.sh if nvcc is unavailable." >&2
        exit 1
    fi
    NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    if [[ "$NVCC_RELEASE" != "12.8" ]]; then
        echo "Expected CUDA toolkit 12.8 for the pinned cu128 stack; nvcc reports ${NVCC_RELEASE:-unknown}." >&2
        exit 1
    fi
    if [[ ! -f "${PROJECT_ROOT}/turbodiffusion/ops/cutlass/include/cutlass/cutlass.h" ]]; then
        echo "CUTLASS submodule is missing. Run: git submodule update --init --recursive" >&2
        exit 1
    fi
fi

"$PYTHON_BIN" -m venv "$VENV_PATH"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

if [[ -n "$OFFLINE_DIR" ]]; then
    PIP_SOURCE=(--no-index --find-links "$OFFLINE_DIR")
else
    PIP_SOURCE=()
    python -m pip install --upgrade pip
fi

if [[ -n "$OFFLINE_DIR" ]]; then
    python -m pip install "${PIP_SOURCE[@]}" torch==2.8.0 torchvision==0.23.0 triton==3.4.0
else
    python -m pip install torch==2.8.0 torchvision==0.23.0 triton==3.4.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

python -m pip install "${PIP_SOURCE[@]}" -r "${PROJECT_ROOT}/requirements/b200/${PROFILE}.txt"

if [[ "$WITH_FLASH_ATTN" -eq 1 ]]; then
    if [[ -n "$OFFLINE_DIR" ]]; then
        python -m pip install "${PIP_SOURCE[@]}" -r "${PROJECT_ROOT}/requirements/b200/flash-attn.txt"
    else
        MAX_JOBS="${MAX_JOBS:-8}" TORCH_CUDA_ARCH_LIST="10.0" \
            python -m pip install -r "${PROJECT_ROOT}/requirements/b200/flash-attn.txt" --no-build-isolation
    fi
fi

if [[ "$PROJECT_MODE" == "editable" ]]; then
    TURBODIFFUSION_CUDA_ARCHS=100 MAX_JOBS="${MAX_JOBS:-8}" \
        python -m pip install -e "$PROJECT_ROOT" --no-build-isolation --no-deps
else
    python -m pip install "${PIP_SOURCE[@]}" turbodiffusion==1.0.0 --no-deps
fi

python -m pip check
python -m pip freeze > "${VENV_PATH}/environment.lock.txt"

cat > "${VENV_PATH}/activate-turbodiffusion.sh" <<EOF
#!/usr/bin/env bash
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/turbodiffusion:\${PYTHONPATH:-}"
EOF
chmod +x "${VENV_PATH}/activate-turbodiffusion.sh"

if [[ "$RUN_VERIFY" -eq 1 ]]; then
    PYTHONPATH="${PROJECT_ROOT}/turbodiffusion${PYTHONPATH:+:$PYTHONPATH}" \
        python "${SCRIPT_DIR}/verify.py" --smoke base
fi

echo
echo "B200 environment installed at: $VENV_PATH"
echo "Activate it with: source ${VENV_PATH}/activate-turbodiffusion.sh"
