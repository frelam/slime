#!/bin/bash
# =============================================================================
# Environment Setup Script for Agentic RL GRPO Training
# =============================================================================
#
# Target: Bare-metal machine with V100 GPUs (or any NVIDIA GPU).
# This script auto-detects the CUDA version and installs all dependencies
# needed to run `examples/agentic_rl_grpo/run.sh`.
#
# Usage:
#   # Full install (sandbox mode: E2B + Hermes/Claude Code harness):
#   bash examples/agentic_rl_grpo/setup_env.sh
#
#   # Minimal install (sglang_loop mode: no Docker/E2B needed):
#   SLIME_SETUP_MODE=minimal bash examples/agentic_rl_grpo/setup_env.sh
#
#   # Custom install prefix:
#   SLIME_VENV_DIR=/path/to/venv bash examples/agentic_rl_grpo/setup_env.sh
#
#   # Use conda instead of venv:
#   SLIME_ENV_TYPE=conda SLIME_CONDA_ENV=slime bash examples/agentic_rl_grpo/setup_env.sh
#
#   # Faster setup (skip SGLang source patches — pip only):
#   SLIME_SGLANG_PIP_ONLY=1 bash examples/agentic_rl_grpo/setup_env.sh
#
#   # Use a specific pip mirror:
#   SLIME_PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple bash examples/agentic_rl_grpo/setup_env.sh
#
#   # Force default PyPI (no mirror):
#   SLIME_PIP_MIRROR="" bash examples/agentic_rl_grpo/setup_env.sh
#
# What this script does:
#   1. Checks system requirements (CUDA, Python, gcc, git)
#   2. Creates a Python virtual environment (venv or conda)
#   3. Installs PyTorch matching the detected CUDA version
#   4. Installs Megatron-LM from NVIDIA repo (with slime patches)
#   5. Installs SGLang (with slime patches for logprob capture)
#   6. Installs slime and all Python dependencies
#   7. Verifies the installation
#
# Requirements (must be pre-installed):
#   - CUDA toolkit 11.8+ (script auto-detects version)
#   - Python 3.10+
#   - gcc/g++ 9+ (for compiling Megatron kernels)
#   - git
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

# "full" = sandbox mode (E2B + harnesses), "minimal" = sglang_loop mode
SETUP_MODE="${SLIME_SETUP_MODE:-full}"

# Environment type: "venv" (Python venv) or "conda"
ENV_TYPE="${SLIME_ENV_TYPE:-venv}"

# Venv directory (for venv mode)
VENV_DIR="${SLIME_VENV_DIR:-$HOME/slime_env}"

# Conda environment name (for conda mode)
CONDA_ENV="${SLIME_CONDA_ENV:-slime}"

# Megatron-LM commit (pinned for reproducibility)
MEGATRON_COMMIT="${SLIME_MEGATRON_COMMIT:-1dcf0dafa884ad52ffb243625717a3471643e087}"
MEGATRON_REPO="${SLIME_MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"

# SGLang version tag (pinned for reproducibility)
SGLANG_VERSION="${SLIME_SGLANG_VERSION:-v0.5.13-cu129}"
SGLANG_REPO="${SLIME_SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"

# Skip SGLang source patches? Set to 1 for faster setup (pip only, no patches).
# Default: apply patches (if available) for full slime compatibility.
SGLANG_PIP_ONLY="${SLIME_SGLANG_PIP_ONLY:-0}"

# Pip mirror for faster downloads. Auto-detect if in China → use Tsinghua.
# Set explicitly: SLIME_PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
# Disable mirror:  SLIME_PIP_MIRROR=""
PIP_MIRROR="${SLIME_PIP_MIRROR:-auto}"

# Patch version (matches docker/patch/<version>/)
PATCH_VERSION="${SLIME_PATCH_VERSION:-latest}"

# Number of parallel build jobs
MAX_JOBS="${SLIME_MAX_JOBS:-16}"

# Path to slime repo (auto-detected)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }
log_step()  { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BLUE}▶${NC} $*"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ---- Pip mirror auto-detection ----

# Chinese mirrors (order: Tsinghua → Aliyun → USTC)
_MIRROR_CANDIDATES=(
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://mirrors.aliyun.com/pypi/simple"
    "https://pypi.mirrors.ustc.edu.cn/simple"
)

resolve_pip_mirror() {
    # Explicitly set
    if [ "$PIP_MIRROR" != "auto" ]; then
        echo "$PIP_MIRROR"
        return
    fi

    # Auto: test if we're in China by checking latency to Baidu DNS
    if timeout 1 bash -c "echo >/dev/tcp/180.101.50.242/53" 2>/dev/null; then
        # Test each mirror, return first reachable
        for mirror in "${_MIRROR_CANDIDATES[@]}"; do
            if curl -s --connect-timeout 3 --max-time 5 -o /dev/null "$mirror" 2>/dev/null; then
                echo "$mirror"
                return
            fi
        done
    fi

    # Fallback: empty (use default PyPI)
    echo ""
}

PIP_MIRROR_URL="$(resolve_pip_mirror)"

if [ -n "$PIP_MIRROR_URL" ]; then
    log_info "Using pip mirror: $PIP_MIRROR_URL"
else
    log_info "Using default PyPI (no mirror configured)"
fi

# pip wrapper that adds mirror flag automatically
_pip() {
    if [ -n "$PIP_MIRROR_URL" ]; then
        pip "$@" -i "$PIP_MIRROR_URL"
    else
        pip "$@"
    fi
}

# ---------------------------------------------------------------------------
# Check system requirements
# ---------------------------------------------------------------------------

check_requirements() {
    log_step "Step 1/7: Checking system requirements..."

    # Check Python
    if ! command -v python3 &>/dev/null; then
        log_error "python3 not found. Please install Python 3.10+ first."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        log_error "Python 3.10+ required, but found Python $PYTHON_VERSION"
        exit 1
    fi
    log_info "Python $PYTHON_VERSION ✓"

    # Check CUDA
    if ! command -v nvcc &>/dev/null; then
        log_warn "nvcc not found in PATH. Searching for CUDA..."
        if [ -d "/usr/local/cuda" ]; then
            export PATH="/usr/local/cuda/bin:$PATH"
            log_info "Found CUDA at /usr/local/cuda"
        elif [ -d "/usr/local/cuda-12" ]; then
            export PATH="/usr/local/cuda-12/bin:$PATH"
            log_info "Found CUDA at /usr/local/cuda-12"
        else
            log_error "CUDA not found. Please install CUDA toolkit 11.8+ first."
            log_error "  Download: https://developer.nvidia.com/cuda-downloads"
            exit 1
        fi
    fi

    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "unknown")
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

    if [ "$CUDA_MAJOR" -lt 11 ] || ([ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -lt 8 ]); then
        log_error "CUDA 11.8+ required, but found CUDA $CUDA_VERSION"
        exit 1
    fi
    log_info "CUDA $CUDA_VERSION ✓"

    # Check GPU
    if command -v nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        log_info "GPU: $GPU_NAME (x$GPU_COUNT) ✓"

        # V100-specific warnings
        if echo "$GPU_NAME" | grep -qi "V100"; then
            log_warn "V100 detected (SM70). Some packages will be skipped:"
            log_warn "  - flash-attn 2/3 (requires SM80+) → using ring_flash_attn instead"
            log_warn "  - transformer_engine (requires SM80+) → skipped"
            log_warn "  - FlashQLA / tilelang (requires SM90+) → skipped"
            IS_V100=1
        else
            IS_V100=0
        fi
    else
        log_warn "nvidia-smi not found. Assuming GPU is available."
        IS_V100=0
    fi

    # Check gcc/g++
    if ! command -v gcc &>/dev/null || ! command -v g++ &>/dev/null; then
        log_error "gcc/g++ not found. Please install build-essential."
        exit 1
    fi
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    if [ "$GCC_VERSION" -lt 9 ]; then
        log_error "gcc 9+ required, but found gcc $GCC_VERSION"
        exit 1
    fi
    log_info "gcc $(gcc -dumpversion) ✓"

    # Check git
    if ! command -v git &>/dev/null; then
        log_error "git not found. Please install git."
        exit 1
    fi
    log_info "git $(git --version | cut -d' ' -f3) ✓"

    # Check disk space (need ~20GB)
    AVAILABLE_GB=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo "0")
    if [ "$AVAILABLE_GB" -lt 20 ]; then
        log_warn "Less than 20GB free disk space (${AVAILABLE_GB}GB). Installation may fail."
    else
        log_info "Disk space: ${AVAILABLE_GB}GB available ✓"
    fi

    log_info "All system requirements met."
}

# ---------------------------------------------------------------------------
# Create Python environment
# ---------------------------------------------------------------------------

create_environment() {
    log_step "Step 2/7: Creating Python environment ($ENV_TYPE)..."

    if [ "$ENV_TYPE" = "conda" ]; then
        if ! command -v conda &>/dev/null; then
            log_error "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi

        if conda env list | grep -q "^${CONDA_ENV} "; then
            log_warn "Conda environment '$CONDA_ENV' already exists."
            read -rp "  Remove and recreate? [y/N] " yn
            if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
                conda env remove -n "$CONDA_ENV" -y
            else
                log_info "Using existing environment."
            fi
        fi

        if ! conda env list | grep -q "^${CONDA_ENV} "; then
            conda create -n "$CONDA_ENV" python=3.10 -y
        fi

        # Activate conda environment
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"

        log_info "Conda environment '$CONDA_ENV' ready ✓"
        PYTHON_BIN="$(which python3)"
    else
        if [ -d "$VENV_DIR" ]; then
            log_warn "Venv directory '$VENV_DIR' already exists."
            read -rp "  Remove and recreate? [y/N] " yn
            if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
                rm -rf "$VENV_DIR"
            else
                log_info "Using existing venv."
            fi
        fi

        if [ ! -d "$VENV_DIR" ]; then
            python3 -m venv "$VENV_DIR"
        fi

        # Activate
        source "$VENV_DIR/bin/activate"

        # Upgrade pip
        _pip install --upgrade pip setuptools wheel

        log_info "Venv '$VENV_DIR' ready ✓"
        PYTHON_BIN="$VENV_DIR/bin/python3"
    fi

    log_info "Python: $($PYTHON_BIN --version)"
    log_info "pip: $(pip --version)"
}

# ---------------------------------------------------------------------------
# Install PyTorch
# ---------------------------------------------------------------------------

install_pytorch() {
    log_step "Step 3/7: Installing PyTorch..."

    # Detect CUDA major.minor for PyTorch index
    CUDA_VER_SHORT="${CUDA_MAJOR}$(printf '%02d' "$CUDA_MINOR" 2>/dev/null || echo "0${CUDA_MINOR}")"

    # Map CUDA version to PyTorch index
    # CUDA 11.8 → cu118, CUDA 12.1 → cu121, CUDA 12.4 → cu124, etc.
    if [ "$CUDA_MAJOR" -eq 11 ]; then
        TORCH_CUDA="cu118"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -le 1 ]; then
        TORCH_CUDA="cu121"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -le 4 ]; then
        TORCH_CUDA="cu124"
    else
        TORCH_CUDA="cu124"  # CUDA 12.4+
    fi

    # For V100, use PyTorch 2.5.x (most stable for SM70)
    # For newer GPUs, use latest stable
    if [ "${IS_V100:-0}" -eq 1 ]; then
        TORCH_VERSION="2.5.1"
        log_info "V100 detected → installing PyTorch $TORCH_VERSION (stable for SM70)"
        TORCH_SPEC="torch==$TORCH_VERSION"
    else
        TORCH_SPEC="torch"
    fi

    # PyTorch CUDA wheels: mirrors like Tsinghua/Aliyun DO host them, so we can
    # use the mirror here directly (much faster in China). On default PyPI we
    # must use the pytorch.org index because PyPI doesn't host CUDA wheels.
    if [ -n "$PIP_MIRROR_URL" ]; then
        log_info "Installing PyTorch via mirror ($PIP_MIRROR_URL)..."
        _pip install "$TORCH_SPEC" torchvision torchaudio
    else
        log_info "Installing PyTorch from pytorch.org ($TORCH_CUDA)..."
        pip install "$TORCH_SPEC" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    fi

    # Verify installation
    python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
"

    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log_error "PyTorch CUDA check failed. Check your CUDA installation."
        log_error "Try: nvidia-smi  and  nvcc --version"
        exit 1
    fi

    log_info "PyTorch installed and verified ✓"
}

# ---------------------------------------------------------------------------
# Install Megatron-LM (with slime patches)
# ---------------------------------------------------------------------------

install_megatron() {
    log_step "Step 4/7: Installing Megatron-LM..."

    MEGATRON_DIR="$HOME/Megatron-LM-slime"

    if [ -d "$MEGATRON_DIR" ]; then
        log_warn "Megatron directory exists: $MEGATRON_DIR"
        read -rp "  Remove and reclone? [y/N] " yn
        if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
            rm -rf "$MEGATRON_DIR"
        fi
    fi

    if [ ! -d "$MEGATRON_DIR" ]; then
        log_info "Cloning Megatron-LM from $MEGATRON_REPO..."
        git clone "$MEGATRON_REPO" "$MEGATRON_DIR" --recursive
        cd "$MEGATRON_DIR"
        git checkout "$MEGATRON_COMMIT"
        log_info "Megatron-LM cloned at commit $MEGATRON_COMMIT"
    fi

    cd "$MEGATRON_DIR"

    # Apply Megatron patch if available
    PATCH_FILE="$SLIME_ROOT/docker/patch/${PATCH_VERSION}/megatron.patch"
    if [ -f "$PATCH_FILE" ]; then
        log_info "Applying Megatron patch: $PATCH_FILE"
        git update-index --refresh 2>/dev/null || true
        if git apply --check "$PATCH_FILE" 2>/dev/null; then
            git apply "$PATCH_FILE" --3way || {
                log_warn "Megatron patch had conflicts. Continuing anyway..."
            }
        else
            log_warn "Megatron patch does not apply cleanly (may already be applied)."
        fi
    else
        log_warn "Megatron patch not found: $PATCH_FILE"
        log_warn "Continuing without patch — some features may not work."
    fi

    # Force numpy < 2 (Megatron requirement)
    _pip install "numpy<2"

    # Install Megatron
    log_info "Installing Megatron-LM..."
    _pip install -e . --no-deps 2>&1 | tail -5

    # Verify
    if python3 -c "import megatron" 2>/dev/null; then
        log_info "Megatron-LM installed ✓"
    else
        log_error "Megatron-LM import failed!"
        log_error "Check: cd $MEGATRON_DIR && pip install -e . --no-deps"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Install SGLang (with slime patches)
# ---------------------------------------------------------------------------
#
# Strategy: pip-install SGLang first to get all compiled CUDA kernels (this is
# fast because they ship as precompiled wheels). Then clone the source at the
# matching version, apply slime patches, and install in editable mode on top.
# Editable install replaces Python code only while keeping the pip-installed
# compiled extensions. This gives us patched code + fast install.
#
# Set SLIME_SGLANG_PIP_ONLY=1 to skip patches and use pip-installed SGLang as-is
# (faster, but may lack logprob/weight-pull fixes needed by slime).

install_sglang() {
    log_step "Step 5/7: Installing SGLang..."

    SGLANG_DIR="$HOME/sglang-slime"

    # ---- Phase 1: pip-install SGLang for compiled kernels ----
    log_info "Installing SGLang via pip (precompiled kernels)..."
    _pip install "sglang[all]" 2>&1 | tail -5 || {
        log_error "SGLang pip install failed."
        exit 1
    }

    # Verify pip install worked
    if ! python3 -c "import sglang" 2>/dev/null; then
        log_error "SGLang import failed after pip install!"
        exit 1
    fi
    SGLANG_VER=$(python3 -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    log_info "SGLang $SGLANG_VER installed via pip ✓"

    # ---- Check if we should skip patching ----
    if [ "${SLIME_SGLANG_PIP_ONLY:-0}" = "1" ]; then
        log_warn "SLIME_SGLANG_PIP_ONLY=1 — skipping patches and source install."
        log_warn "Some slime features (weight-pull, logprob format) may not work correctly."
        return 0
    fi

    # Check if patches are even available
    PATCH_DIR="$SLIME_ROOT/docker/patch/${PATCH_VERSION}"
    HAS_PATCHES=0
    for p in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
        if [ -f "$PATCH_DIR/$p" ]; then
            HAS_PATCHES=1
            break
        fi
    done

    if [ "$HAS_PATCHES" -eq 0 ]; then
        log_warn "No SGLang patches found in $PATCH_DIR — using pip-installed SGLang as-is."
        return 0
    fi

    # ---- Phase 2: Clone source, apply patches, editable install ----
    if [ -d "$SGLANG_DIR" ]; then
        log_warn "SGLang source directory exists: $SGLANG_DIR"
        read -rp "  Remove and reclone? [y/N] " yn
        if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
            rm -rf "$SGLANG_DIR"
        fi
    fi

    if [ ! -d "$SGLANG_DIR" ]; then
        log_info "Cloning SGLang source from $SGLANG_REPO..."
        # Shallow clone to save time/space
        git clone --depth 1 --branch "$SGLANG_VERSION" "$SGLANG_REPO" "$SGLANG_DIR" 2>/dev/null || {
            log_warn "Tag $SGLANG_VERSION not found, cloning main..."
            git clone --depth 1 "$SGLANG_REPO" "$SGLANG_DIR" 2>/dev/null || {
                log_warn "Clone failed. Using pip-installed SGLang as-is."
                return 0
            }
        }
        log_info "SGLang source cloned"
    fi

    cd "$SGLANG_DIR"

    # Apply SGLang patches — these are critical for:
    #   sglang.patch              — general compatibility with slime
    #   sglang-top_p.patch        — top_p sampling correctness
    #   sglang-release_hicache.patch — release KV cache on weight update
    #   sglang-pull_weights.patch — pull_weights() for direct tensor sync
    for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
        PATCH_FILE="$PATCH_DIR/$patch_name"
        if [ -f "$PATCH_FILE" ]; then
            log_info "Applying $patch_name..."
            git update-index --refresh 2>/dev/null || true
            if git apply --check "$PATCH_FILE" 2>/dev/null; then
                git apply "$PATCH_FILE" --3way || log_warn "  $patch_name had conflicts"
            else
                log_warn "  $patch_name does not apply cleanly (may already be applied or incompatible with this SGLang version)"
            fi
        fi
    done

    # Editable install (replaces Python code, keeps pip-installed compiled extensions)
    log_info "Installing patched SGLang in editable mode..."
    _pip install -e "python" --no-deps --no-build-isolation 2>&1 | tail -5 || {
        log_warn "SGLang editable install failed. Pip-installed version will be used."
    }

    # Final verification
    if python3 -c "import sglang" 2>/dev/null; then
        SGLANG_VER=$(python3 -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        log_info "SGLang $SGLANG_VER with patches ✓"
    else
        log_error "SGLang import failed after patching!"
        log_error "Reinstalling SGLang via pip as fallback..."
        _pip install "sglang[all]" --force-reinstall 2>&1 | tail -3
        log_info "SGLang reinstalled via pip (without patches)"
    fi
}

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------

install_python_deps() {
    log_step "Step 6/7: Installing Python dependencies..."

    # ---- Base dependencies (always needed) ----
    log_info "Installing base dependencies..."

    _pip install \
        pyyaml \
        safetensors \
        transformers \
        datasets \
        accelerate \
        tensorboard \
        wandb \
        ray[default] \
        xxhash \
        zstandard \
        aiohttp \
        httpx \
        numba \
        omegaconf \
        pillow \
        pylatexenc \
        blake3 \
        blobfile \
        qwen_vl_utils

    # ---- ring_flash_attn (replaces flash-attn on V100) ----
    log_info "Installing ring_flash_attn..."
    _pip install ring_flash_attn 2>/dev/null || {
        log_warn "ring_flash_attn install failed. Continuing..."
    }

    # ---- flash-attn (skip on V100 — requires SM80+) ----
    if [ "${IS_V100:-0}" -eq 1 ]; then
        log_warn "Skipping flash-attn (requires SM80+, V100 is SM70)"
    else
        log_info "Installing flash-attn..."
        MAX_JOBS="$MAX_JOBS" _pip install flash-attn --no-build-isolation 2>/dev/null || {
            log_warn "flash-attn install failed. Using ring_flash_attn fallback."
        }
    fi

    # ---- TransformerEngine (skip on V100 — requires SM80+) ----
    if [ "${IS_V100:-0}" -eq 1 ]; then
        log_warn "Skipping transformer_engine (requires SM80+, V100 is SM70)"
    else
        log_info "Installing transformer_engine..."
        _pip install "transformer_engine[pytorch]" --no-build-isolation 2>/dev/null || {
            log_warn "transformer_engine install failed. Continuing..."
        }
    fi

    # ---- APEX (optional, skip on V100 if compilation fails) ----
    if [ "${IS_V100:-0}" -eq 1 ]; then
        log_warn "Skipping APEX install on V100 (often fails on SM70)"
    else
        log_info "Installing APEX..."
        if [ ! -d "$HOME/apex" ]; then
            git clone https://github.com/NVIDIA/apex.git "$HOME/apex" --depth 1 2>/dev/null || true
        fi
        if [ -d "$HOME/apex" ]; then
            cd "$HOME/apex"
            _pip install -v --disable-pip-version-check --no-cache-dir \
                --no-build-isolation \
                --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
                . 2>&1 | tail -5 || log_warn "APEX install failed. Continuing..."
        fi
    fi

    # ---- SGLang Router (for weight sync) ----
    log_info "Installing sglang-router..."
    _pip install "sglang-router>=0.2.3" 2>/dev/null || {
        log_warn "sglang-router from PyPI failed. Trying GitHub release..."
        pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl 2>/dev/null || {
            log_warn "sglang-router wheel failed. Continuing..."
        }
    }

    # ---- megatron-bridge (model-specific weight conversion) ----
    log_info "Installing megatron-bridge..."
    pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation 2>/dev/null || {
        log_warn "megatron-bridge install failed. Continuing..."
    }

    # ---- mbridge (model bridging utilities) ----
    log_info "Installing mbridge..."
    pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps 2>/dev/null || {
        log_warn "mbridge install failed. Continuing..."
    }

    # ---- torch_memory_saver (GPU memory optimization) ----
    log_info "Installing torch_memory_saver..."
    TMS_CUDA_MAJOR="${CUDA_MAJOR:-12}"
    pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir 2>/dev/null || {
        log_warn "torch_memory_saver install failed. Continuing..."
    }

    # ---- nvidia-modelopt (optional, for model optimization) ----
    _pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation 2>/dev/null || {
        log_warn "nvidia-modelopt install failed. Continuing..."
    }

    # ---- emerging-optimizers (Muon optimizer for GRPO) ----
    log_info "Installing emerging-optimizers (Muon optimizer)..."
    _pip install emerging-optimizers 2>/dev/null || {
        log_warn "emerging-optimizers install failed. Muon optimizer won't be available."
    }

    # ---- sgl_kernel (optional, for SGLang performance) ----
    _pip install sgl-kernel 2>/dev/null || {
        log_warn "sgl-kernel install failed. Continuing..."
    }

    # ---- Full mode: E2B, API clients, swebench ----
    if [ "$SETUP_MODE" = "full" ]; then
        log_info "Installing full-mode dependencies (sandbox + harnesses)..."

        _pip install \
            e2b \
            anthropic \
            openai \
            "openai-agents" \
            "mcp[cli]" \
            memray \
            docker

        # swebench (for SWE-bench evaluation)
        _pip install swebench 2>/dev/null || {
            log_warn "swebench install failed. SWE-bench evaluation won't work."
        }

        log_info "Full-mode dependencies installed ✓"
    else
        log_info "Minimal mode: skipping E2B, API clients, swebench, memray, docker"
    fi

    # ---- Install slime itself ----
    log_info "Installing slime from $SLIME_ROOT..."
    cd "$SLIME_ROOT"
    _pip install -e . --no-deps

    # Install int4_qat kernels if available
    if [ -f "$SLIME_ROOT/slime/backends/megatron_utils/kernels/int4_qat/setup.py" ]; then
        log_info "Installing int4_qat kernels..."
        cd "$SLIME_ROOT/slime/backends/megatron_utils/kernels/int4_qat"
        _pip install . --no-build-isolation 2>/dev/null || log_warn "int4_qat install failed."
    fi

    # Verify slime installation
    if python3 -c "import slime" 2>/dev/null; then
        log_info "slime installed ✓"
    else
        log_error "slime import failed!"
        exit 1
    fi

    log_info "All Python dependencies installed ✓"
}

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------

verify_installation() {
    log_step "Step 7/7: Verifying installation..."

    python3 -c "
import importlib
import sys

modules = [
    ('torch', 'PyTorch'),
    ('megatron', 'Megatron-LM'),
    ('sglang', 'SGLang'),
    ('slime', 'slime'),
    ('transformers', 'Transformers'),
    ('ray', 'Ray'),
    ('safetensors', 'SafeTensors'),
    ('datasets', 'Datasets'),
    ('wandb', 'W&B'),
    ('tensorboard', 'TensorBoard'),
    ('aiohttp', 'aiohttp'),
]

print()
print('=' * 60)
print('  Installation Verification')
print('=' * 60)

all_ok = True
for mod, name in modules:
    try:
        importlib.import_module(mod)
        ver = getattr(sys.modules[mod], '__version__', '?')
        print(f'  ✓ {name:25s} {ver}')
    except ImportError:
        print(f'  ✗ {name:25s} NOT FOUND')
        all_ok = False

# Optional modules
optionals = [
    ('ring_flash_attn', 'Ring Flash Attn'),
    ('apex', 'APEX'),
    ('transformer_engine', 'TransformerEngine'),
    ('e2b', 'E2B Sandbox'),
    ('anthropic', 'Anthropic SDK'),
    ('openai', 'OpenAI SDK'),
    ('swebench', 'SWE-bench'),
    ('emerging_optimizers', 'Muon Optimizer'),
]
print()
print('  Optional:')
for mod, name in optionals:
    try:
        importlib.import_module(mod)
        ver = getattr(sys.modules[mod], '__version__', '?')
        print(f'  ✓ {name:25s} {ver}')
    except ImportError:
        print(f'  - {name:25s} (not installed)')

print()
if all_ok:
    print('  ✓ All core modules OK')
else:
    print('  ✗ Some core modules missing — check logs above')
print('=' * 60)
"

    # Check GPU visibility
    log_info "GPU check:"
    python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)')
else:
    print('  WARNING: No CUDA devices visible!')
"

    # Print paths
    log_info "Installation complete. Summary:"
    echo "  Python:   $(which python3)"
    echo "  pip:      $(which pip)"
    echo "  slime:    $SLIME_ROOT"
    echo "  Megatron: $HOME/Megatron-LM-slime"
    echo "  SGLang:   $HOME/sglang-slime"
    echo "  Mode:     $SETUP_MODE"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Setup complete! Next steps:"
    echo ""
    if [ "$ENV_TYPE" = "conda" ]; then
        echo "  1. Activate:  conda activate $CONDA_ENV"
    else
        echo "  1. Activate:  source $VENV_DIR/bin/activate"
    fi
    echo "  2. Run:       bash examples/agentic_rl_grpo/run.sh"
    echo ""
    if [ "$SETUP_MODE" = "minimal" ]; then
        echo "  Note: Using sglang_loop mode (no Docker/E2B needed)."
        echo "        Set: export SLIME_AGENT_MODE=sglang_loop"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║   slime Agentic RL GRPO — Environment Setup                 ║"
    echo "║   Mode: $SETUP_MODE                                          ║"
    if [ -n "$PIP_MIRROR_URL" ]; then
        echo "║   Mirror: $PIP_MIRROR_URL"
    fi
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    check_requirements
    create_environment
    install_pytorch
    install_megatron
    install_sglang
    install_python_deps
    verify_installation
}

# Run
main "$@"
