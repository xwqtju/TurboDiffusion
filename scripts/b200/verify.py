#!/usr/bin/env python3
"""Deterministic environment and CUDA smoke checks for the B200 deployment."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


def version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def add_import(results: list[CheckResult], module: str, fatal: bool = True) -> None:
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "version unavailable")
        results.append(CheckResult(f"import {module}", True, str(version), fatal))
    except Exception as exc:  # noqa: BLE001 - diagnostics should report every import failure
        results.append(CheckResult(f"import {module}", False, repr(exc), fatal))


def run_smoke(torch, mode: str, results: list[CheckResult]) -> None:
    if mode == "none" or not torch.cuda.is_available():
        return

    try:
        a = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        results.append(CheckResult("BF16 GEMM", bool(torch.isfinite(c).all()), str(tuple(c.shape))))
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("BF16 GEMM", False, repr(exc)))

    try:
        q = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        results.append(CheckResult("cuDNN/SDPA attention", bool(torch.isfinite(out).all()), str(tuple(out.shape))))
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("cuDNN/SDPA attention", False, repr(exc)))

    try:
        from ops.core import int8_linear, int8_quant

        x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
        w_q, w_scale = int8_quant(w)
        out = int8_linear(x, w_q, w_scale)
        torch.cuda.synchronize()
        results.append(CheckResult("TurboDiffusion INT8 CUDA extension", bool(torch.isfinite(out).all()), str(tuple(out.shape))))
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("TurboDiffusion INT8 CUDA extension", False, repr(exc)))

    if mode == "all":
        try:
            from SLA import SparseLinearAttention

            module = SparseLinearAttention(head_dim=128, topk=1.0, BLKQ=128, BLKK=64).cuda().eval()
            q = torch.randn((1, 128, 1, 128), device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            with torch.no_grad():
                out = module(q, k, v)
            torch.cuda.synchronize()
            results.append(CheckResult("SLA Triton forward", bool(torch.isfinite(out).all()), str(tuple(out.shape))))
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult("SLA Triton forward", False, repr(exc)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", choices=("none", "base", "all"), default="base")
    parser.add_argument("--allow-non-b200", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    source_root = project_root / "turbodiffusion"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    results: list[CheckResult] = []
    py_ok = sys.version_info[:2] == (3, 11)
    results.append(CheckResult("Python 3.11", py_ok, sys.version.split()[0]))

    try:
        import torch

        results.append(CheckResult("PyTorch 2.8.0", torch.__version__.startswith("2.8.0"), torch.__version__))
        cuda_version = torch.version.cuda or "not available"
        results.append(CheckResult("PyTorch CUDA >= 12.8", version_tuple(cuda_version) >= (12, 8), cuda_version))
        results.append(CheckResult("CUDA available", torch.cuda.is_available(), str(torch.cuda.is_available())))

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            is_b200 = capability == (10, 0) and "B200" in name.upper()
            results.append(
                CheckResult(
                    "B200 / SM100",
                    is_b200 or args.allow_non_b200,
                    f"{name}; compute capability {capability[0]}.{capability[1]}",
                )
            )
        run_smoke(torch, args.smoke, results)
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("import torch", False, repr(exc)))

    for module in ("triton", "einops", "transformers", "turbo_diffusion_ops"):
        add_import(results, module)
    add_import(results, "flash_attn", fatal=False)
    add_import(results, "spas_sage_attn", fatal=False)

    nvcc = shutil.which("nvcc")
    if nvcc:
        proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False)
        detail = next((line.strip() for line in proc.stdout.splitlines() if "release" in line), proc.stdout.strip())
        results.append(CheckResult("nvcc present", proc.returncode == 0, detail, fatal=False))
    else:
        results.append(CheckResult("nvcc present", False, "not found; prebuilt-wheel runtime is still usable", fatal=False))

    ffmpeg = shutil.which("ffmpeg")
    results.append(CheckResult("ffmpeg executable", ffmpeg is not None, ffmpeg or "not found", fatal=False))

    print("TurboDiffusion B200 environment report")
    print(f"PYTHONPATH source root: {source_root}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print()
    for result in results:
        if result.ok:
            status = "PASS"
        elif result.fatal:
            status = "FAIL"
        else:
            status = "WARN"
        print(f"[{status:4}] {result.name}: {result.detail}")

    fatal_failures = [result for result in results if result.fatal and not result.ok]
    if fatal_failures:
        print(f"\nEnvironment is NOT ready ({len(fatal_failures)} fatal check(s) failed).")
        return 1

    print("\nEnvironment is ready for the supported B200 paths: original and SLA.")
    print("SageSLA/SpargeAttn is intentionally not required because upstream does not support SM100.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
