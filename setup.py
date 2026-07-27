""" 
Copyright (c) 2025 by TurboDiffusion team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
  author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
"""

import os
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ops_dir = Path(__file__).parent / "turbodiffusion" / "ops"
cutlass_dir = ops_dir / "cutlass"

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=--verbose,--warn-on-local-memory-usage",
    "-lineinfo",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-Xcompiler",
    "-fPIC"
]

_SUPPORTED_CUDA_ARCHS = {
    "80": ("compute_80", "sm_80"),
    "89": ("compute_89", "sm_89"),
    "90": ("compute_90", "sm_90"),
    "100": ("compute_100", "sm_100"),
    "120a": ("compute_120a", "sm_120a"),
}


def get_cuda_arch_flags():
    """Return explicit CUDA targets, optionally narrowed for deployment builds.

    TURBODIFFUSION_CUDA_ARCHS accepts a comma/semicolon/space separated list,
    for example ``100`` for an NVIDIA B200-only build. Keeping the historical
    multi-architecture default preserves the behavior of release builds.
    """

    requested = os.environ.get("TURBODIFFUSION_CUDA_ARCHS")
    archs = ["120a", "100", "90", "89", "80"]
    if requested:
        archs = requested.replace(",", " ").replace(";", " ").split()
        unknown = sorted(set(archs) - _SUPPORTED_CUDA_ARCHS.keys())
        if unknown:
            supported = ", ".join(_SUPPORTED_CUDA_ARCHS)
            raise RuntimeError(
                f"Unsupported TURBODIFFUSION_CUDA_ARCHS value(s): {unknown}. "
                f"Supported values: {supported}."
            )

    flags = []
    for arch in archs:
        compute, code = _SUPPORTED_CUDA_ARCHS[arch]
        flags.extend(["-gencode", f"arch={compute},code={code}"])
    return flags


cc_flag = get_cuda_arch_flags()

ext_modules = [
    CUDAExtension(
        name="turbo_diffusion_ops",
        sources=[
            "turbodiffusion/ops/bindings.cpp",
            "turbodiffusion/ops/quant/quant.cu", 
            "turbodiffusion/ops/norm/rmsnorm.cu",
            "turbodiffusion/ops/norm/layernorm.cu",
            "turbodiffusion/ops/gemm/gemm.cu"
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags + ["-DEXECMODE=0"] + cc_flag + ["--threads", "4"],
        },
        include_dirs=[
            cutlass_dir / "include",
            cutlass_dir / "tools" / "util" / "include",
            ops_dir 
        ],
        libraries=["cuda"],
    )
]

setup(
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks")
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
