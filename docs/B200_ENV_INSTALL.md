# B200 TurboDiffusion environment installation

This guide is for a company NVIDIA B200 machine where the `turbodiffusion`
Python environment already exists but PyTorch and the project dependencies are
not installed yet.

The commands below intentionally use pinned versions from this repository,
instead of the latest packages, so that the B200 environment is reproducible.

## 1. Target baseline

| Component | Required baseline |
|---|---|
| GPU | NVIDIA B200, compute capability 10.0 / SM100 |
| OS | Linux x86_64 |
| Python | 3.11 |
| CUDA driver | CUDA 12.8-compatible or newer data-center driver |
| CUDA toolkit | 12.8, required when building TurboDiffusion CUDA ops from source |
| PyTorch | 2.8.0 cu128 |
| torchvision | 0.23.0 cu128 |
| Triton | 3.4.0 |
| TurboDiffusion CUDA arch | SM100 only |

Check the host first:

```bash
nvidia-smi
python --version
nvcc --version
uname -m
```

Expected:

- `python --version` reports Python 3.11.x.
- `uname -m` reports `x86_64`.
- `nvcc --version` reports CUDA 12.8 if you plan to install the project in
  editable/source mode.
- The B200 GPU is visible in `nvidia-smi`.

If the machine only has the NVIDIA driver and no CUDA toolkit, use a prebuilt
TurboDiffusion wheel. Editable/source installation needs `nvcc`.

## 2. Activate the existing environment

If the environment was created by conda:

```bash
conda activate turbodiffusion
```

If the environment was created by `venv`, replace the path with the actual one:

```bash
source /path/to/turbodiffusion/bin/activate
```

Make sure the activated Python is 3.11:

```bash
python -c "import sys; print(sys.version)"
```

Upgrade the basic packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. Prepare the repository

Enter the TurboDiffusion repository:

```bash
cd /path/to/TurboDiffusion
```

Initialize CUTLASS before building the project CUDA extension:

```bash
git submodule update --init --recursive
```

Confirm CUTLASS exists:

```bash
test -f turbodiffusion/ops/cutlass/include/cutlass/cutlass.h
```

If this command fails, the submodule was not initialized correctly.

## 4. Install PyTorch for B200

Install the CUDA 12.8 PyTorch stack:

```bash
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  triton==3.4.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify PyTorch sees the B200:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

Expected capability is `(10, 0)`.

## 5. Install runtime dependencies

For inference only:

```bash
python -m pip install -r requirements/b200/runtime.txt
```

For training, QAT experiments, or SLA fine-tuning:

```bash
python -m pip install -r requirements/b200/training.txt
```

`training.txt` includes the runtime dependency file, so do not install both
unless you simply want to be explicit.

## 6. Install TurboDiffusion

Recommended source/editable install on the B200:

```bash
TURBODIFFUSION_CUDA_ARCHS=100 \
MAX_JOBS="${MAX_JOBS:-8}" \
python -m pip install -e . --no-build-isolation --no-deps
```

Why these flags matter:

- `TURBODIFFUSION_CUDA_ARCHS=100` builds only SM100 kernels for B200.
- `--no-build-isolation` makes the build use the already-installed PyTorch.
- `--no-deps` prevents pip from replacing the pinned dependency stack.

Then set `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd)/turbodiffusion:${PYTHONPATH:-}"
```

Run this `export PYTHONPATH=...` command each time after activating the
environment, unless your shell startup script or job launcher already sets it.

## 7. Do not install SageSLA on B200

Use:

```text
--attention_type sla
```

or:

```text
--attention_type original
```

Do not use:

```text
--attention_type sagesla
```

SageSLA depends on SpargeAttn kernels. The current B200 profile in this
repository treats SpargeAttn as unsupported on SM100, so `sla` is the supported
sparse-linear attention path for B200.

FlashAttention is optional for this B200 path. The repository has a PyTorch
RoPE fallback for the supported `sla` and `original` modes. Only try
FlashAttention after the basic environment passes verification:

```bash
MAX_JOBS="${MAX_JOBS:-8}" TORCH_CUDA_ARCH_LIST="10.0" \
python -m pip install -r requirements/b200/flash-attn.txt --no-build-isolation
```

## 8. Verify the environment

Run the base smoke test first:

```bash
python scripts/b200/verify.py --smoke base
```

Then run all B200 smoke tests:

```bash
python scripts/b200/verify.py --smoke all
```

The base smoke test checks:

- PyTorch CUDA availability
- B200 compute capability
- BF16 GEMM
- cuDNN SDPA
- TurboDiffusion custom INT8 extension import and execution

The full smoke test also compiles/runs the SLA Triton forward path.

Save an environment lock file after a successful install:

```bash
python -m pip freeze > b200-environment.lock.txt
python -m pip check
```

Keep `b200-environment.lock.txt` with experiment logs.

## 9. Prepare checkpoints

For the first inference test, use the unquantized 1.3B checkpoint:

```text
checkpoints/
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
└── TurboWan2.1-T2V-1.3B-480P.pth
```

Use the checkpoint without `-quant` first. Add quantized checkpoints and
`--quant_linear` only after the unquantized path works.

## 10. First inference run

The B200 wrapper is the easiest first run:

```bash
export DIT_PATH=/models/TurboWan2.1-T2V-1.3B-480P.pth
export VAE_PATH=/models/Wan2.1_VAE.pth
export TEXT_ENCODER_PATH=/models/models_t5_umt5-xxl-enc-bf16.pth
export PROMPT='A cinematic tracking shot of a sailboat crossing calm water at sunrise.'
export SAVE_PATH=/output/b200-smoke.mp4

scripts/b200/run_t2v_1_3b.sh
```

The wrapper defaults to:

```text
ATTENTION_TYPE=sla
SLA_TOPK=0.1
RESOLUTION=480p
ASPECT_RATIO=16:9
NUM_FRAMES=81
NUM_SAMPLES=1
NUM_STEPS=4
SEED=0
LINEAR_Q_2TO4=0
```

To isolate whether an error comes from SLA/Triton:

```bash
ATTENTION_TYPE=original scripts/b200/run_t2v_1_3b.sh
```

To enable the first-version Rubin-style simulation, which only zeros Q in the
linear-attention branch:

```bash
LINEAR_Q_2TO4=1 scripts/b200/run_t2v_1_3b.sh
```

This is a functional/quality simulation only. It does not provide real Rubin
activation-sparse MMA or SpMM speedup on B200.

## 11. Manual inference command

If you do not use the wrapper:

```bash
export PYTHONPATH="$(pwd)/turbodiffusion:${PYTHONPATH:-}"

python turbodiffusion/inference/wan2.1_t2v_infer.py \
  --model Wan2.1-1.3B \
  --dit_path /models/TurboWan2.1-T2V-1.3B-480P.pth \
  --vae_path /models/Wan2.1_VAE.pth \
  --text_encoder_path /models/models_t5_umt5-xxl-enc-bf16.pth \
  --prompt "A cinematic tracking shot of a sailboat crossing calm water at sunrise." \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --num_frames 81 \
  --num_samples 1 \
  --num_steps 4 \
  --seed 0 \
  --attention_type sla \
  --sla_topk 0.1 \
  --save_path output/b200-smoke.mp4
```

Add `--linear_q_2to4` to enable Q 2:4 activation sparsity simulation.

## 12. Alternative: create a fresh environment with the repo installer

If you can discard the empty environment and create a fresh one, the repository
installer performs the same steps:

```bash
cd /path/to/TurboDiffusion
git submodule update --init --recursive

scripts/b200/install.sh --profile runtime
source .venv-b200/activate-turbodiffusion.sh
python scripts/b200/verify.py --smoke all
```

For training:

```bash
scripts/b200/install.sh --profile training
source .venv-b200/activate-turbodiffusion.sh
python scripts/b200/verify.py --smoke all
```

The installer refuses to reuse an existing environment path. This is deliberate:
it avoids silently mixing old and new dependency stacks.

## 13. Daily sync with the remote main repository

The B200 machine can use the helper script to frequently sync with the remote
main repository and push local changes back:

```bash
scripts/b200/sync_main.sh -m "Update B200 experiment config"
```

The script does the following:

1. Ensures `origin` points to `https://github.com/xwqtju/TurboDiffusion.git`.
2. Fetches `origin/main`.
3. Commits local tracked-file changes.
4. Rebases the local `main` branch on top of `origin/main`.
5. Pushes the result to `origin/main`.

By default it only commits changes to files already tracked by Git. This avoids
accidentally pushing checkpoints, generated videos, logs, or local PDFs.

To include new files, pass `--include-untracked` and preferably restrict the
path range:

```bash
scripts/b200/sync_main.sh \
  --include-untracked \
  -m "Add B200 helper scripts" \
  -- docs scripts/b200
```

Untracked files larger than 10 MiB are rejected unless
`--allow-large-files` is explicitly passed.

To check what the script would do without changing the repository:

```bash
scripts/b200/sync_main.sh --dry-run
```

To commit and rebase locally but skip pushing:

```bash
scripts/b200/sync_main.sh --no-push -m "Local B200 sync"
```

If a rebase conflict happens, the script stops. Resolve the conflicted files,
then run:

```bash
git rebase --continue
scripts/b200/sync_main.sh
```

## 14. Offline installation

Build the wheelhouse on a networked Linux x86_64 host with Python 3.11 and CUDA
12.8:

```bash
cd /path/to/TurboDiffusion
git submodule update --init --recursive
scripts/b200/build_offline_bundle.sh /data/td-b200-bundle --profile training
```

Transfer `/data/td-b200-bundle` and the repository to the company network.
On the B200 machine:

```bash
cd /path/to/td-b200-bundle/wheelhouse
sha256sum -c SHA256SUMS

cd /path/to/TurboDiffusion
scripts/b200/install.sh \
  --offline /path/to/td-b200-bundle/wheelhouse \
  --project wheel \
  --profile training
```

Use `--project editable` instead of `--project wheel` only if the B200 host has
CUDA 12.8 `nvcc` and the CUTLASS submodule initialized.

## 15. Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False` | GPU not exposed to the job/container | Check scheduler GPU allocation, container flags, and `nvidia-smi` |
| Capability is not `(10, 0)` | Not running on B200 | Check `CUDA_VISIBLE_DEVICES` and scheduler allocation |
| `nvcc` not found | CUDA toolkit is not installed | Use prebuilt wheel mode, or install CUDA toolkit 12.8 |
| `no kernel image is available` | CUDA ops were built for the wrong arch | Rebuild with `TURBODIFFUSION_CUDA_ARCHS=100` |
| `turbo_diffusion_ops` import fails | Project extension did not build or wrong torch/CUDA stack | Reinstall after PyTorch cu128 is installed |
| `spas_sage_attn` missing | Expected on B200 | Use `--attention_type sla`, not `sagesla` |
| SLA smoke fails but original attention works | Triton/cache/toolchain issue | Keep logs, then retry after clearing the user Triton cache |
| OOM during install | Too many parallel compile jobs | Lower `MAX_JOBS`, for example `MAX_JOBS=4` |

Avoid adding random CUDA library paths to `LD_LIBRARY_PATH` after installing the
cu128 PyTorch wheels. Mixed CUDA runtimes can fail in confusing ways.
