# TurboDiffusion deployment on NVIDIA B200

This runbook is intended for a company B200 host where interactive debugging or
AI coding is unavailable. It fixes the software baseline, makes CUDA extension
builds target SM100 only, supports an offline wheelhouse, and provides smoke
tests that do not require model checkpoints.

If a `turbodiffusion` Python environment already exists on the B200 host but
PyTorch and the project dependencies are not installed yet, use
[B200_ENV_INSTALL.md](B200_ENV_INSTALL.md) as the step-by-step installation
guide.

## Scope and important architecture limits

- NVIDIA B200 is a Blackwell **SM100** GPU. It is not a Rubin GPU.
- B200 can run the existing model, SLA training, and a software simulation of
  2:4 pruning/QAT. It cannot validate Rubin activation-sparse MMA instructions
  or Rubin speedup.
- Use `--attention_type sla` on B200 for the TurboDiffusion sparse path.
- Use `--attention_type original` as the correctness/debugging baseline.
- Do not use `--attention_type sagesla` on B200. As checked on 2026-07-27,
  upstream SpargeAttn declares only SM80/86/87/89/90 and has no SM100 kernels.
- Start with the unquantized checkpoint on B200. The custom INT8 extension has
  an SM100 build target and is smoke-tested by `verify.py`, but it should be
  enabled only after the unquantized path passes end-to-end inference.

The project uses a pure PyTorch RoPE fallback when FlashAttention is absent.
`flash-attn` is therefore optional for the supported B200 `sla` and `original`
paths, which removes one fragile native dependency from an offline deployment.

## Fixed baseline

| Component | Baseline |
|---|---|
| OS | Linux x86_64 |
| Python | 3.11 |
| NVIDIA driver | CUDA 12.8-compatible data-center driver |
| CUDA toolkit for source builds | 12.8, including `nvcc` |
| PyTorch | 2.8.0 cu128 |
| torchvision | 0.23.0 cu128 |
| Triton | 3.4.0 |
| GPU target | SM100 only |

The CUDA driver may be newer than the toolkit. The toolkit used to compile the
extension must remain 12.8 for this profile. A target with only the NVIDIA
driver can use a prebuilt project wheel.

Before installation, confirm the host basics:

```bash
nvidia-smi
nvcc --version
python3.11 --version
uname -m
```

Expected GPU compute capability is `10.0`. The post-install verifier checks it.

## Online installation on the B200 host

Initialize the pinned CUTLASS submodule before copying the repository, or on a
networked checkout:

```bash
git submodule update --init --recursive
```

Runtime/inference environment:

```bash
scripts/b200/install.sh --profile runtime
source .venv-b200/activate-turbodiffusion.sh
python scripts/b200/verify.py --smoke all
```

Training environment for SLA or future 2:4 sparsity-aware fine-tuning:

```bash
scripts/b200/install.sh --profile training
source .venv-b200/activate-turbodiffusion.sh
python scripts/b200/verify.py --smoke all
```

To try the optional FlashAttention RoPE implementation, add
`--with-flash-attn`. It is not required for the initial deployment.

The installer writes the resolved package set to:

```text
.venv-b200/environment.lock.txt
```

Archive that file with every successful run so the environment is auditable.

## Offline installation

The wheelhouse must be built on a **networked Linux x86_64 host with Python
3.11 and CUDA toolkit 12.8**. Do not build it on macOS: CUDA and Python wheels
are platform-specific.

On the networked build host:

```bash
git submodule update --init --recursive
scripts/b200/build_offline_bundle.sh /data/td-b200-bundle --profile training
```

This creates:

```text
/data/td-b200-bundle/
├── BUNDLE_INFO.txt
├── build-environment.txt
└── wheelhouse/
    ├── SHA256SUMS
    ├── torch/torchvision/triton wheels
    ├── Python dependency wheels
    └── turbodiffusion-1.0.0-*.whl
```

Transfer both the repository and `td-b200-bundle` into the company network.
On the B200 host:

```bash
cd /path/to/td-b200-bundle/wheelhouse
sha256sum -c SHA256SUMS

cd /path/to/TurboDiffusion
scripts/b200/install.sh \
  --offline /path/to/td-b200-bundle/wheelhouse \
  --project wheel \
  --profile training
```

The prebuilt-wheel mode needs only a compatible NVIDIA driver at runtime. To
modify and rebuild CUDA code on the B200 host, use `--project editable`; that
mode additionally requires CUDA 12.8 `nvcc` and the initialized CUTLASS
submodule.

## Checkpoints and datasets

Checkpoints are intentionally separate from the Python wheelhouse because they
are large and have a different update lifecycle. For the first B200 test, copy:

```text
checkpoints/
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
└── TurboWan2.1-T2V-1.3B-480P.pth
```

Use the checkpoint without the `-quant` suffix for the baseline.

For training, the published WebDataset is downloaded separately:

```bash
git clone https://huggingface.co/datasets/worstcoder/Wan_datasets assets/datasets
```

The supplied SLA configuration expects:

```text
assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K/shard*.tar
```

Each sample contains `latent.pt`, `embed.pt`, and `prompt.txt`. Copy the dataset
and checkpoint files with checksums generated by the company's approved file
transfer process.

## First end-to-end run

Use the wrapper so paths and the B200-safe attention backend are explicit:

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
attention_type=sla
sla_topk=0.1
linear_q_2to4=false
resolution=480p
num_frames=81
num_steps=4
```

To simulate Rubin-style 2:4 activation sparsity on Q only in the SLA linear
attention branch, run:

```bash
LINEAR_Q_2TO4=1 scripts/b200/run_t2v_1_3b.sh
```

To isolate an SLA/Triton issue, rerun with:

```bash
ATTENTION_TYPE=original scripts/b200/run_t2v_1_3b.sh
```

## Verification order and failure isolation

Always debug in this order:

1. `python scripts/b200/verify.py --smoke base`
2. `python scripts/b200/verify.py --smoke all`
3. End-to-end inference with `ATTENTION_TYPE=original`
4. End-to-end inference with `ATTENTION_TYPE=sla`
5. Optional quantized checkpoint plus `--quant_linear`

`--smoke base` checks BF16 GEMM, cuDNN SDPA, and the project INT8 extension.
`--smoke all` additionally compiles and runs the SLA Triton forward kernel.

Interpret common failures as follows:

| Failure | Likely cause | Action |
|---|---|---|
| Capability is not 10.0 | Wrong GPU visibility | Check scheduler allocation and `CUDA_VISIBLE_DEVICES` |
| PyTorch reports CUDA unavailable | Driver/container device mapping | Check `nvidia-smi` and container GPU flags |
| `turbo_diffusion_ops` import fails | Missing/incompatible project wheel | Rebuild with Python 3.11, torch 2.8 cu128, CUDA 12.8, target `100` |
| `no kernel image` | Extension was not built for SM100 | Rebuild with `TURBODIFFUSION_CUDA_ARCHS=100` |
| SDPA fails but BF16 GEMM passes | cuDNN/runtime mismatch | Confirm the torch cu128 wheel and CUDA libraries are not shadowed |
| SLA smoke fails | Triton cache/toolchain issue | Clear only the user Triton cache after preserving logs, then rerun; use `original` meanwhile |
| `spas_sage_attn` missing | Expected on B200 | Use `sla`, not `sagesla` |

Do not install or preload a different system CUDA library into the venv. In
particular, avoid adding arbitrary CUDA `lib64` directories to `LD_LIBRARY_PATH`
after installing the cu128 PyTorch wheels; that can silently mix runtimes.

## Relation to Rubin 2:4 work

This environment is suitable for implementing and training a fake 2:4 Q mask
with STE, comparing it against the dense teacher, and measuring quality on
B200. Keep the training forward representation separate from the eventual
Rubin compressed format.

The following cannot be signed off on B200:

- Rubin sparse-MMA instruction encoding and metadata layout;
- actual activation-compression overhead;
- sparse Tensor Core throughput;
- final fused attention kernel speedup.

Those require a Rubin GPU and its matching CUDA/CUTLASS toolchain. Treat B200
results as functional and quality validation, not Rubin performance evidence.
